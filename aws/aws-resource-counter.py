#!/usr/bin/env python3

import boto3
import csv
import logging
import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from botocore.config import Config
from tqdm import tqdm

# =========================
# Global Settings
# =========================

LOG_FILE = "aws_resource_count.log"
CSV_FILE = "aws_resource_counts.csv"
ASSUME_ROLE_NAME = "OrganizationAccountAccessRole"

MAX_ACCOUNT_WORKERS = 32   # processes (accounts in parallel)
MAX_REGION_WORKERS = 10    # threads per account (regions in parallel)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

boto_config = Config(
    retries={"max_attempts": 10, "mode": "standard"},
    connect_timeout=5,
    read_timeout=60,
)


# =========================
# Basic Session / STS Helpers
# =========================

def create_management_session(access_key: str, secret_key: str) -> boto3.Session:
    """Create a boto3 session for the management/current account."""
    return boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def get_account_id(session: boto3.Session) -> str:
    sts = session.client("sts", config=boto_config)
    return sts.get_caller_identity()["Account"]


def get_credentials_tuple(session: boto3.Session):
    """Extract (access_key, secret_key, session_token) for passing to subprocesses."""
    creds = session.get_credentials()
    frozen = creds.get_frozen_credentials()
    return frozen.access_key, frozen.secret_key, frozen.token


def assume_role_into_account(account_id: str, mgmt_creds) -> Optional[boto3.Session]:
    """
    Assume OrganizationAccountAccessRole in a member account.
    mgmt_creds = (access_key, secret_key, session_token) of management account.
    """
    access_key, secret_key, session_token = mgmt_creds

    base_session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token
    )

    sts = base_session.client("sts", config=boto_config)
    role_arn = f"arn:aws:iam::{account_id}:role/{ASSUME_ROLE_NAME}"

    try:
        resp = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="OrgResourceScan"
        )
        c = resp["Credentials"]
        return boto3.Session(
            aws_access_key_id=c["AccessKeyId"],
            aws_secret_access_key=c["SecretAccessKey"],
            aws_session_token=c["SessionToken"],
        )
    except Exception as e:
        logging.error(f"[{account_id}] AssumeRole failed: {e}")
        return None


# =========================
# Global Region List
# =========================

def get_global_region_list() -> List[str]:
    """Get list of all enabled EC2 regions once (used by all workers)."""
    ec2 = boto3.client("ec2", region_name="us-east-1", config=boto_config)
    regions = ec2.describe_regions(AllRegions=False)["Regions"]
    return [r["RegionName"] for r in regions]


REGIONS = get_global_region_list()


# =========================
# Fast "Empty Account" Check
# =========================

def is_account_empty(session: boto3.Session) -> bool:
    """
    Quick check: does the account have *any* tagged resources?
    Uses Resource Groups Tagging API with a single-page request.
    If we can’t tell (AccessDenied, etc.), return False (assume non-empty).
    """
    try:
        client = session.client(
            "resource-groups-tagging-api",
            region_name="us-east-1",
            config=boto_config
        )
        resp = client.get_resources(ResourcesPerPage=1)
        resources = resp.get("ResourceTagMappingList", [])
        if not resources:
            logging.info("Account appears empty (no tagged resources) – skipping full scan.")
            return True
        return False
    except Exception as e:
        logging.warning(f"Could not determine account emptiness (treating as non-empty): {e}")
        return False


# =========================
# Per-Region Resource Counters
# =========================

def count_ec2_instances(session: boto3.Session, region: str) -> int:
    """Count all EC2 instances (any state) in a region."""
    try:
        ec2 = session.client("ec2", region_name=region, config=boto_config)
        paginator = ec2.get_paginator("describe_instances")

        total = 0
        for page in paginator.paginate():
            for res in page.get("Reservations", []):
                total += len(res.get("Instances", []))
        return total
    except Exception as e:
        logging.error(f"[{region}] EC2 count failed: {e}")
        return 0


def count_lambda_functions(session: boto3.Session, region: str) -> int:
    """Count Lambda functions in a region."""
    try:
        lam = session.client("lambda", region_name=region, config=boto_config)
        paginator = lam.get_paginator("list_functions")

        total = 0
        for page in paginator.paginate():
            total += len(page.get("Functions", []))
        return total
    except Exception as e:
        logging.error(f"[{region}] Lambda count failed: {e}")
        return 0


def count_ecs_fargate_tasks(session: boto3.Session, region: str) -> int:
    """
    Count ECS Fargate tasks in a region.
    Uses list_clusters + list_tasks(launchType='FARGATE').
    """
    try:
        ecs = session.client("ecs", region_name=region, config=boto_config)
        cluster_paginator = ecs.get_paginator("list_clusters")

        total = 0
        for c_page in cluster_paginator.paginate():
            for cluster_arn in c_page.get("clusterArns", []):
                task_paginator = ecs.get_paginator("list_tasks")
                for t_page in task_paginator.paginate(cluster=cluster_arn, launchType="FARGATE"):
                    total += len(t_page.get("taskArns", []))
        return total
    except Exception as e:
        logging.error(f"[{region}] ECS Fargate task count failed: {e}")
        return 0


def count_eks_clusters(session: boto3.Session, region: str) -> int:
    """Count EKS clusters in a region."""
    try:
        eks = session.client("eks", region_name=region, config=boto_config)
        paginator = eks.get_paginator("list_clusters")

        total = 0
        for page in paginator.paginate():
            total += len(page.get("clusters", []))
        return total
    except Exception as e:
        logging.error(f"[{region}] EKS cluster count failed: {e}")
        return 0


def count_eks_nodes(session: boto3.Session, region: str) -> int:
    """
    Count unique EKS worker nodes (EC2 instances) in a region using Kubernetes tags:
      kubernetes.io/cluster/<cluster-name> = owned/shared
    """
    try:
        eks = session.client("eks", region_name=region, config=boto_config)
        ec2 = session.client("ec2", region_name=region, config=boto_config)

        nodes = set()

        cluster_paginator = eks.get_paginator("list_clusters")
        for c_page in cluster_paginator.paginate():
            for cluster_name in c_page.get("clusters", []):
                inst_paginator = ec2.get_paginator("describe_instances")
                filters = [
                    {
                        "Name": f"tag:kubernetes.io/cluster/{cluster_name}",
                        "Values": ["owned", "shared"]
                    }
                ]
                for i_page in inst_paginator.paginate(Filters=filters):
                    for res in i_page.get("Reservations", []):
                        for inst in res.get("Instances", []):
                            if inst.get("State", {}).get("Name") == "running":
                                nodes.add(inst["InstanceId"])

        return len(nodes)
    except Exception as e:
        logging.error(f"[{region}] EKS node count failed: {e}")
        return 0


def count_ecr_repositories(session: boto3.Session, region: str) -> int:
    """Count ECR repositories in a region."""
    try:
        ecr = session.client("ecr", region_name=region, config=boto_config)
        paginator = ecr.get_paginator("describe_repositories")

        total = 0
        for page in paginator.paginate():
            total += len(page.get("repositories", []))
        return total
    except Exception as e:
        logging.error(f"[{region}] ECR repo count failed: {e}")
        return 0


def count_ecr_images(session: boto3.Session, region: str) -> int:
    """Count ECR images in all repositories in a region."""
    try:
        ecr = session.client("ecr", region_name=region, config=boto_config)
        repo_paginator = ecr.get_paginator("describe_repositories")

        total_images = 0
        for r_page in repo_paginator.paginate():
            for repo in r_page.get("repositories", []):
                name = repo["repositoryName"]
                img_paginator = ecr.get_paginator("describe_images")
                for i_page in img_paginator.paginate(repositoryName=name):
                    total_images += len(i_page.get("imageDetails", []))
        return total_images
    except Exception as e:
        logging.error(f"[{region}] ECR image count failed: {e}")
        return 0


# =========================
# Region Dispatcher (per Account)
# =========================

def count_resources_in_region(session: boto3.Session, region: str) -> Dict[str, int]:
    """
    Count all desired resources in a single region.
    Uses threads to parallelize per-service calls inside the region.
    """
    with ThreadPoolExecutor(max_workers=MAX_REGION_WORKERS) as executor:
        future_ec2 = executor.submit(count_ec2_instances, session, region)
        future_lambda = executor.submit(count_lambda_functions, session, region)
        future_ecs_fargate = executor.submit(count_ecs_fargate_tasks, session, region)
        future_eks_clusters = executor.submit(count_eks_clusters, session, region)
        future_eks_nodes = executor.submit(count_eks_nodes, session, region)
        future_ecr_repos = executor.submit(count_ecr_repositories, session, region)
        future_ecr_images = executor.submit(count_ecr_images, session, region)

        return {
            "ec2": future_ec2.result(),
            "lambda": future_lambda.result(),
            "ecs_fargate": future_ecs_fargate.result(),
            "eks_clusters": future_eks_clusters.result(),
            "eks_nodes": future_eks_nodes.result(),
            "ecr_repos": future_ecr_repos.result(),
            "ecr_images": future_ecr_images.result(),
        }


# =========================
# Per-Account Aggregation
# =========================

def aggregate_account_resources(session: boto3.Session, account_id: str) -> Dict[str, Any]:
    """
    Scan all regions in an account (using threads per region) and aggregate totals.
    """
    totals = {
        "AccountID": account_id,
        "EC2": 0,
        "Lambda": 0,
        "ECS_Fargate": 0,
        "EKS_Clusters": 0,
        "EKS_Nodes": 0,
        "ECR_Repos": 0,
        "ECR_Images": 0,
    }

    for region in REGIONS:
        counts = count_resources_in_region(session, region)
        totals["EC2"] += counts["ec2"]
        totals["Lambda"] += counts["lambda"]
        totals["ECS_Fargate"] += counts["ecs_fargate"]
        totals["EKS_Clusters"] += counts["eks_clusters"]
        totals["EKS_Nodes"] += counts["eks_nodes"]
        totals["ECR_Repos"] += counts["ecr_repos"]
        totals["ECR_Images"] += counts["ecr_images"]

    logging.info(
        f"[{account_id}] EC2={totals['EC2']} "
        f"Lambda={totals['Lambda']} ECS_Fargate={totals['ECS_Fargate']} "
        f"EKS_Clusters={totals['EKS_Clusters']} EKS_Nodes={totals['EKS_Nodes']} "
        f"ECR_Repos={totals['ECR_Repos']} ECR_Images={totals['ECR_Images']}"
    )

    return totals


# =========================
# Worker for ProcessPool (ORG Mode)
# =========================

def scan_member_account_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Executed in a separate process for each member account.
    Payload contains: { "account_id", "mgmt_creds" }
    """
    account_id = payload["account_id"]
    mgmt_creds = payload["mgmt_creds"]

    # Protect STS throttling slightly
    time.sleep(random.uniform(0.05, 0.25))

    session = assume_role_into_account(account_id, mgmt_creds)
    if not session:
        return None

    # Skip empty accounts fast
    if is_account_empty(session):
        logging.info(f"[{account_id}] Skipped (empty account).")
        return {
            "AccountID": account_id,
            "EC2": 0,
            "Lambda": 0,
            "ECS_Fargate": 0,
            "EKS_Clusters": 0,
            "EKS_Nodes": 0,
            "ECR_Repos": 0,
            "ECR_Images": 0,
        }

    return aggregate_account_resources(session, account_id)


# =========================
# CSV Helpers
# =========================

def init_csv(filename: str) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Account ID",
            "EC2 Instances",
            "Lambda Functions",
            "ECS Fargate Tasks",
            "EKS Clusters",
            "EKS Nodes",
            "ECR Repositories",
            "ECR Images",
        ])


def append_account_to_csv(filename: str, row: Dict[str, Any]) -> None:
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["AccountID"],
            row["EC2"],
            row["Lambda"],
            row["ECS_Fargate"],
            row["EKS_Clusters"],
            row["EKS_Nodes"],
            row["ECR_Repos"],
            row["ECR_Images"],
        ])


# =========================
# ORG Mode
# =========================

def list_org_member_accounts(mgmt_session: boto3.Session) -> List[str]:
    org = mgmt_session.client("organizations", config=boto_config)
    account_ids: List[str] = []
    paginator = org.get_paginator("list_accounts")
    for page in paginator.paginate():
        for acct in page.get("Accounts", []):
            if acct.get("Status") == "ACTIVE":
                account_ids.append(acct["Id"])
    return account_ids


def run_org_mode(mgmt_session: boto3.Session) -> None:
    mgmt_account_id = get_account_id(mgmt_session)
    logging.info(f"Management account: {mgmt_account_id}")

    mgmt_creds = get_credentials_tuple(mgmt_session)
    member_account_ids = list_org_member_accounts(mgmt_session)
    logging.info(f"Found {len(member_account_ids)} ACTIVE member accounts.")

    init_csv(CSV_FILE)

    # 1) Scan management account itself (synchronously, same as ACCOUNT mode)
    logging.info(f"Scanning management account {mgmt_account_id}")
    if is_account_empty(mgmt_session):
        logging.info(f"[{mgmt_account_id}] Skipped (empty management account).")
        mgmt_totals = {
            "AccountID": mgmt_account_id,
            "EC2": 0,
            "Lambda": 0,
            "ECS_Fargate": 0,
            "EKS_Clusters": 0,
            "EKS_Nodes": 0,
            "ECR_Repos": 0,
            "ECR_Images": 0,
        }
    else:
        mgmt_totals = aggregate_account_resources(mgmt_session, mgmt_account_id)

    append_account_to_csv(CSV_FILE, mgmt_totals)

    # 2) Scan member accounts in parallel (ProcessPool)
    tasks = []
    for acc_id in member_account_ids:
        tasks.append({"account_id": acc_id, "mgmt_creds": mgmt_creds})

    logging.info("Starting parallel scan of member accounts...")

    with ProcessPoolExecutor(max_workers=MAX_ACCOUNT_WORKERS) as executor:
        future_to_account = {
            executor.submit(scan_member_account_worker, payload): payload["account_id"]
            for payload in tasks
        }

        for future in tqdm(as_completed(future_to_account), total=len(future_to_account), desc="Member accounts"):
            account_id = future_to_account[future]
            try:
                result = future.result()
                if result:
                    append_account_to_csv(CSV_FILE, result)
            except Exception as e:
                logging.error(f"[{account_id}] Error in worker: {e}")

    logging.info(f"ORG mode scan complete. Results in {CSV_FILE}")


# =========================
# ACCOUNT Mode
# =========================

def run_account_mode(mgmt_session: boto3.Session) -> None:
    account_id = get_account_id(mgmt_session)
    logging.info(f"Running ACCOUNT mode for account {account_id}")

    init_csv(CSV_FILE)

    if is_account_empty(mgmt_session):
        logging.info(f"[{account_id}] Skipped (empty account).")
        totals = {
            "AccountID": account_id,
            "EC2": 0,
            "Lambda": 0,
            "ECS_Fargate": 0,
            "EKS_Clusters": 0,
            "EKS_Nodes": 0,
            "ECR_Repos": 0,
            "ECR_Images": 0,
        }
    else:
        totals = aggregate_account_resources(mgmt_session, account_id)

    append_account_to_csv(CSV_FILE, totals)
    logging.info(f"ACCOUNT mode scan complete. Results in {CSV_FILE}")


# =========================
# MAIN
# =========================

def main():
    print("Enter 'org' for organization-level resource counting")
    print("Enter 'account' for single-account resource counting")
    input_type = input("Your choice (org/account): ").strip().lower()

    if input_type not in ("org", "account"):
        print("Invalid input. Please enter 'org' or 'account'.")
        logging.error("Invalid mode selected.")
        return

    access_key = input("Enter the access key for the management/current account: ").strip()
    secret_key = input("Enter the secret key for the management/current account: ").strip()

    mgmt_session = create_management_session(access_key, secret_key)

    if input_type == "org":
        run_org_mode(mgmt_session)
    else:
        run_account_mode(mgmt_session)

    print(f"Done. Results written to {CSV_FILE}")
    print(f"Logs written to {LOG_FILE}")


if __name__ == "__main__":
    main()
