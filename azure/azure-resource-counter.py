#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import logging
import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from tqdm import tqdm

from azure.identity import ClientSecretCredential
from azure.core.exceptions import HttpResponseError
from azure.mgmt.subscription import SubscriptionClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resourcegraph import ResourceGraphClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.web import WebSiteManagementClient
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerservice import ContainerServiceClient
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azure.containerregistry import ContainerRegistryClient

# ================================================================
# CONFIG
# ================================================================

LOG_FILE = "azure_resource_scan.log"
CSV_FILE = "azure_resource_counts.csv"

# Safer concurrency to avoid heavy throttling
MAX_SUBSCRIPTION_PROCESSES = 10   # subscriptions in parallel
MAX_RG_THREADS = 8                # resource groups per subscription in parallel
MAX_RESOURCE_THREADS = 4          # per-RG resource counters in parallel

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

console = logging.getLogger("console")
console.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))
console.addHandler(ch)


# ================================================================
# ARM THROTTLING WRAPPER
# ================================================================

def arm_call_with_throttle(handler, *args, **kwargs):
    """
    Adaptive throttling wrapper for ARM SDK calls.
    Retries on SubscriptionRequestsThrottled / Retry-After with exponential backoff + jitter.
    """
    retries = 0
    max_retries = 8

    while True:
        try:
            return handler(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "SubscriptionRequestsThrottled" in msg or "Retry-After" in msg:
                wait = min(2 ** retries + random.uniform(0, 1.5), 20)
                logging.warning(f"ARM throttled. Retrying in {wait:.1f} seconds...")
                time.sleep(wait)
                retries += 1
                if retries > max_retries:
                    logging.error("Maximum ARM throttle retries exceeded. Giving up on this call.")
                    # Let caller handle this as a failure
                    raise
                continue
            # Not a throttling error; let caller handle
            raise


# ================================================================
# AUTH
# ================================================================

def authenticate(client_id: str, client_secret: str, tenant_id: str) -> ClientSecretCredential:
    return ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)


# ================================================================
# SUBSCRIPTIONS
# ================================================================

def list_subscriptions(credential: ClientSecretCredential) -> List[str]:
    sub_client = SubscriptionClient(credential)
    subs = [s.subscription_id for s in sub_client.subscriptions.list()]
    logging.info(f"Found {len(subs)} subscriptions.")
    return subs


# ================================================================
# SKIP-EMPTY-SUBSCRIPTION CHECK (SAFE)
# ================================================================

def subscription_is_empty(credential: ClientSecretCredential, subscription_id: str) -> bool:
    """
    Uses Azure Resource Graph to see if subscription has ANY resources.
    Uses 'limit 1' to avoid schema/aggregation issues.
    If anything weird happens, treat as NON-empty (avoid false skips).
    """
    try:
        rg_client = ResourceGraphClient(credential)

        query = "Resources | limit 1"

        result = arm_call_with_throttle(
            rg_client.resources,
            query=query,
            subscriptions=[subscription_id]
        )

        data = getattr(result, "data", None)

        # Completely empty list -> clearly empty
        if isinstance(data, list) and len(data) == 0:
            logging.info(f"[{subscription_id}] Subscription appears EMPTY (no resources).")
            return True

        # List with at least one row -> not empty
        if isinstance(data, list) and len(data) > 0:
            return False

        # Returned a string or something odd: treat as NOT empty
        if isinstance(data, str):
            logging.warning(
                f"[{subscription_id}] Resource Graph returned string data; "
                f"treating subscription as NON-empty."
            )
            return False

        # Unknown shape → safe default: not empty
        return False

    except Exception as e:
        logging.warning(
            f"[{subscription_id}] Unable to check emptiness via Resource Graph, "
            f"treating as NON-empty. Error: {e}"
        )
        return False


# ================================================================
# RESOURCE COUNTERS (PER RESOURCE GROUP)
# ================================================================

def count_vms(cred: ClientSecretCredential, sub_id: str, rg: str) -> int:
    try:
        client = ComputeManagementClient(cred, sub_id)
        vm_list = arm_call_with_throttle(client.virtual_machines.list, rg)
        return sum(1 for _ in vm_list)
    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(f"[{sub_id}] No permission to list VMs in RG {rg} (need Reader on compute).")
        else:
            logging.error(f"[{sub_id}] Error counting VMs in {rg}: {e}")
        return 0
    except Exception as e:
        logging.error(f"[{sub_id}] Error counting VMs in {rg}: {e}")
        return 0


def count_webapps(cred: ClientSecretCredential, sub_id: str, rg: str) -> int:
    try:
        client = WebSiteManagementClient(cred, sub_id)
        apps = arm_call_with_throttle(client.web_apps.list_by_resource_group, rg)
        return sum(1 for _ in apps)
    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(f"[{sub_id}] No permission to list WebApps in RG {rg} (need Reader on web).")
        else:
            logging.error(f"[{sub_id}] Error counting WebApps in {rg}: {e}")
        return 0
    except Exception as e:
        logging.error(f"[{sub_id}] Error counting WebApps in {rg}: {e}")
        return 0


def count_container_instances(cred: ClientSecretCredential, sub_id: str, rg: str) -> int:
    try:
        client = ContainerInstanceManagementClient(cred, sub_id)
        groups = arm_call_with_throttle(client.container_groups.list_by_resource_group, rg)
        return sum(1 for _ in groups)
    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(
                f"[{sub_id}] No permission to list Container Instances in RG {rg} "
                f"(need Reader on Microsoft.ContainerInstance)."
            )
        else:
            logging.error(f"[{sub_id}] Error counting Container Instances in {rg}: {e}")
        return 0
    except Exception as e:
        logging.error(f"[{sub_id}] Error counting Container Instances in {rg}: {e}")
        return 0


def count_aks_clusters_and_nodes(
    cred: ClientSecretCredential,
    sub_id: str,
    rg: str
) -> Dict[str, int]:
    """
    Count AKS clusters + AKS nodes:
      - AKS_Clusters: number of managed clusters
      - AKS_Nodes_Desired: sum of agent pool 'count'
      - AKS_Nodes_Actual: count of underlying VMSS instances tagged to the cluster
    """
    clusters = 0
    nodes_desired = 0
    nodes_actual = 0

    try:
        aks_client = ContainerServiceClient(cred, sub_id)
        compute_client = ComputeManagementClient(cred, sub_id)

        # 1) List managed clusters in RG
        aks_list = list(
            arm_call_with_throttle(
                aks_client.managed_clusters.list_by_resource_group, rg
            )
        )
        clusters = len(aks_list)

        for mc in aks_list:
            # Desired node count: sum of all agent pool profile counts
            profiles = getattr(mc, "agent_pool_profiles", None)
            if profiles:
                for p in profiles:
                    if getattr(p, "count", None) is not None:
                        nodes_desired += p.count

            # Actual node count: count VMSS instances tagged with this cluster
            cluster_name = mc.name

            try:
                vmss_list = list(
                    arm_call_with_throttle(
                        compute_client.virtual_machine_scale_sets.list, rg
                    )
                )
            except HttpResponseError as e:
                if e.status_code == 403:
                    logging.warning(
                        f"[{sub_id}] No permission to list VMSS (for AKS nodes) in RG {rg}; "
                        f"actual node count may be zero."
                    )
                    continue
                logging.error(f"[{sub_id}] Error listing VMSS in RG {rg}: {e}")
                continue
            except Exception as e:
                logging.error(f"[{sub_id}] Error listing VMSS in RG {rg}: {e}")
                continue

            for vmss in vmss_list:
                tags = getattr(vmss, "tags", {}) or {}
                cluster_tag = tags.get("kubernetes.azure.com/clusterName")
                if cluster_tag and cluster_tag.lower() == cluster_name.lower():
                    try:
                        insts = arm_call_with_throttle(
                            compute_client.virtual_machine_scale_set_vms.list,
                            rg,
                            vmss.name
                        )
                        for _ in insts:
                            nodes_actual += 1
                    except HttpResponseError as e:
                        if e.status_code == 403:
                            logging.warning(
                                f"[{sub_id}] No permission to list VMSS VMs for {vmss.name}; "
                                f"actual node count may be incomplete."
                            )
                        else:
                            logging.error(
                                f"[{sub_id}] Error listing VMSS VMs in {rg} "
                                f"for {vmss.name}: {e}"
                            )
                    except Exception as e:
                        logging.error(
                            f"[{sub_id}] Error listing VMSS VMs in {rg} "
                            f"for {vmss.name}: {e}"
                        )

    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(
                f"[{sub_id}] No permission to list AKS clusters in RG {rg} "
                f"(need Reader on Microsoft.ContainerService/managedClusters)."
            )
        else:
            logging.error(f"[{sub_id}] Error counting AKS clusters in {rg}: {e}")
    except Exception as e:
        logging.error(f"[{sub_id}] Error counting AKS clusters in {rg}: {e}")

    return {
        "clusters": clusters,
        "nodes_desired": nodes_desired,
        "nodes_actual": nodes_actual
    }


def count_acr_registries(cred: ClientSecretCredential, sub_id: str, rg: str) -> int:
    try:
        client = ContainerRegistryManagementClient(cred, sub_id)
        regs = arm_call_with_throttle(client.registries.list_by_resource_group, rg)
        return sum(1 for _ in regs)
    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(
                f"[{sub_id}] No permission to list ACR registries in RG {rg} "
                f"(need Reader on Microsoft.ContainerRegistry/registries)."
            )
        else:
            logging.error(f"[{sub_id}] Error counting ACR registries in {rg}: {e}")
        return 0
    except Exception as e:
        logging.error(f"[{sub_id}] Error counting ACR registries in {rg}: {e}")
        return 0


def count_acr_images_fast(cred: ClientSecretCredential, sub_id: str, rg: str) -> int:
    """
    Count ACR images using the data-plane SDK (ContainerRegistryClient).
    If data-plane access is missing (AcrPull/AcrReader), we log a warning.
    """
    total = 0
    try:
        mgmt_client = ContainerRegistryManagementClient(cred, sub_id)
        registries = list(
            arm_call_with_throttle(
                mgmt_client.registries.list_by_resource_group, rg
            )
        )

        for registry in registries:
            login_server = registry.login_server  # e.g. myregistry.azurecr.io
            endpoint = f"https://{login_server}"

            try:
                data_client = ContainerRegistryClient(endpoint, cred)
                repos = data_client.list_repository_names()
                for repo in repos:
                    manifests = data_client.list_manifest_properties(repo)
                    total += sum(1 for _ in manifests)
            except HttpResponseError as e:
                if e.status_code in (401, 403):
                    logging.warning(
                        f"[{sub_id}] No data-plane access to ACR '{registry.name}' "
                        f"(need AcrPull/AcrReader/AcrContributor). "
                        f"ACR image counts will be 0 for this registry."
                    )
                else:
                    logging.error(
                        f"[{sub_id}] Error listing ACR images in registry {registry.name}: {e}"
                    )
            except Exception as e:
                logging.error(
                    f"[{sub_id}] General error listing ACR images in registry "
                    f"{registry.name}: {e}"
                )

    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(
                f"[{sub_id}] No ARM permission to enumerate ACR registries in RG {rg} "
                f"(need Reader on Microsoft.ContainerRegistry/registries)."
            )
        else:
            logging.error(f"[{sub_id}] Error enumerating ACR registries in {rg}: {e}")
        return 0
    except Exception as e:
        logging.error(f"[{sub_id}] Error enumerating ACR registries in {rg}: {e}")
        return 0

    return total


def count_functions(cred: ClientSecretCredential, sub_id: str, rg: str) -> int:
    try:
        client = WebSiteManagementClient(cred, sub_id)
        apps = arm_call_with_throttle(client.web_apps.list_by_resource_group, rg)
        return sum(
            1 for app in apps
            if app.kind and "functionapp" in app.kind.lower()
        )
    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(
                f"[{sub_id}] No permission to list Functions in RG {rg} "
                f"(need Reader on Microsoft.Web/sites)."
            )
        else:
            logging.error(f"[{sub_id}] Error counting Functions in {rg}: {e}")
        return 0
    except Exception as e:
        logging.error(f"[{sub_id}] Error counting Functions in {rg}: {e}")
        return 0


# ================================================================
# PER-RG RESOURCE AGGREGATOR
# ================================================================

def count_rg_resources(
    cred: ClientSecretCredential,
    sub_id: str,
    rg_name: str
) -> Dict[str, int]:
    """
    Count all relevant resources in a single resource group, using threads.
    Returns a dict that can be aggregated at subscription level.
    """
    results = {
        "VMs": 0,
        "WebApps": 0,
        "ACI": 0,
        "AKS_Clusters": 0,
        "AKS_Nodes_Desired": 0,
        "AKS_Nodes_Actual": 0,
        "ACR_Registries": 0,
        "ACR_Images": 0,
        "Functions": 0,
    }

    with ThreadPoolExecutor(max_workers=MAX_RESOURCE_THREADS) as executor:
        futures = {
            "VMs": executor.submit(count_vms, cred, sub_id, rg_name),
            "WebApps": executor.submit(count_webapps, cred, sub_id, rg_name),
            "ACI": executor.submit(count_container_instances, cred, sub_id, rg_name),
            "AKS": executor.submit(count_aks_clusters_and_nodes, cred, sub_id, rg_name),
            "ACR_Registries": executor.submit(count_acr_registries, cred, sub_id, rg_name),
            "ACR_Images": executor.submit(count_acr_images_fast, cred, sub_id, rg_name),
            "Functions": executor.submit(count_functions, cred, sub_id, rg_name),
        }

        for key, fut in futures.items():
            value = fut.result()
            if key == "AKS":
                results["AKS_Clusters"] += value["clusters"]
                results["AKS_Nodes_Desired"] += value["nodes_desired"]
                results["AKS_Nodes_Actual"] += value["nodes_actual"]
            elif key == "VMs":
                results["VMs"] += value
            elif key == "WebApps":
                results["WebApps"] += value
            elif key == "ACI":
                results["ACI"] += value
            elif key == "ACR_Registries":
                results["ACR_Registries"] += value
            elif key == "ACR_Images":
                results["ACR_Images"] += value
            elif key == "Functions":
                results["Functions"] += value

    return results


# ================================================================
# PER-SUBSCRIPTION WORKER (Process)
# ================================================================

def process_subscription(
    sub_id: str,
    client_id: str,
    client_secret: str,
    tenant_id: str
) -> Optional[Dict[str, Any]]:
    """
    Entry point for a subscription worker process.
    Creates its own credential, optionally skips empty subs, and aggregates counts.
    """
    cred = authenticate(client_id, client_secret, tenant_id)

    # Empty subscription optimization
    if subscription_is_empty(cred, sub_id):
        return {
            "Subscription": sub_id,
            "VMs": 0,
            "WebApps": 0,
            "ACI": 0,
            "AKS_Clusters": 0,
            "AKS_Nodes_Desired": 0,
            "AKS_Nodes_Actual": 0,
            "ACR_Registries": 0,
            "ACR_Images": 0,
            "Functions": 0,
        }

    rg_client = ResourceManagementClient(cred, sub_id)

    try:
        rg_iter = arm_call_with_throttle(rg_client.resource_groups.list)
        rgs = list(rg_iter)
    except HttpResponseError as e:
        if e.status_code == 403:
            logging.warning(
                f"[{sub_id}] No permission to list resource groups; "
                f"subscription may show zeros for all counts."
            )
        else:
            logging.error(f"[{sub_id}] Error listing resource groups: {e}")
        return None
    except Exception as e:
        logging.error(f"[{sub_id}] Error listing resource groups: {e}")
        return None

    totals = {
        "Subscription": sub_id,
        "VMs": 0,
        "WebApps": 0,
        "ACI": 0,
        "AKS_Clusters": 0,
        "AKS_Nodes_Desired": 0,
        "AKS_Nodes_Actual": 0,
        "ACR_Registries": 0,
        "ACR_Images": 0,
        "Functions": 0,
    }

    with ThreadPoolExecutor(max_workers=MAX_RG_THREADS) as executor:
        future_to_rg = {
            executor.submit(count_rg_resources, cred, sub_id, rg.name): rg.name
            for rg in rgs
        }

        for fut in as_completed(future_to_rg):
            rg_name = future_to_rg[fut]
            try:
                res = fut.result()
                totals["VMs"] += res["VMs"]
                totals["WebApps"] += res["WebApps"]
                totals["ACI"] += res["ACI"]
                totals["AKS_Clusters"] += res["AKS_Clusters"]
                totals["AKS_Nodes_Desired"] += res["AKS_Nodes_Desired"]
                totals["AKS_Nodes_Actual"] += res["AKS_Nodes_Actual"]
                totals["ACR_Registries"] += res["ACR_Registries"]
                totals["ACR_Images"] += res["ACR_Images"]
                totals["Functions"] += res["Functions"]
            except Exception as e:
                logging.error(f"[{sub_id}] Error processing RG {rg_name}: {e}")

    logging.info(
        f"[{sub_id}] Totals: "
        f"VMs={totals['VMs']}, WebApps={totals['WebApps']}, ACI={totals['ACI']}, "
        f"AKS_Clusters={totals['AKS_Clusters']}, AKS_Nodes_Desired={totals['AKS_Nodes_Desired']}, "
        f"AKS_Nodes_Actual={totals['AKS_Nodes_Actual']}, "
        f"ACR_Registries={totals['ACR_Registries']}, ACR_Images={totals['ACR_Images']}, "
        f"Functions={totals['Functions']}"
    )

    return totals


# ================================================================
# CSV HELPERS
# ================================================================

def write_csv_header() -> None:
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Subscription ID",
            "Virtual Machines",
            "Web Apps",
            "Container Instances",
            "AKS Clusters",
            "AKS Nodes (desired)",
            "AKS Nodes (actual)",
            "ACR Registries",
            "ACR Images",
            "Azure Functions",
        ])


def append_csv(row: Dict[str, Any]) -> None:
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["Subscription"],
            row["VMs"],
            row["WebApps"],
            row["ACI"],
            row["AKS_Clusters"],
            row["AKS_Nodes_Desired"],
            row["AKS_Nodes_Actual"],
            row["ACR_Registries"],
            row["ACR_Images"],
            row["Functions"],
        ])


# ================================================================
# MAIN
# ================================================================

def main():
    console.info("\nAzure Resource Scanner (Multi-Subscription)")
    console.info("===========================================")
    console.info("1. Scan ALL subscriptions")
    console.info("2. Scan SINGLE subscription")
    mode = input("Choose 1/2: ").strip()

    client_id = input("Client ID: ").strip()
    client_secret = input("Client Secret: ").strip()
    tenant_id = input("Tenant ID: ").strip()

    cred = authenticate(client_id, client_secret, tenant_id)

    if mode == "1":
        subs = list_subscriptions(cred)

        write_csv_header()

        with ProcessPoolExecutor(max_workers=MAX_SUBSCRIPTION_PROCESSES) as executor:
            futures = [
                executor.submit(process_subscription, sub_id, client_id, client_secret, tenant_id)
                for sub_id in subs
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Subscriptions"):
                result = fut.result()
                if result:
                    append_csv(result)

        console.info(f"\n✅ Done. Output: {CSV_FILE}")
        console.info(f"📝 Logs: {LOG_FILE}")

    elif mode == "2":
        sub_id = input("Enter Subscription ID: ").strip()
        write_csv_header()
        result = process_subscription(sub_id, client_id, client_secret, tenant_id)
        if result:
            append_csv(result)
        console.info(f"\n✅ Done. Output: {CSV_FILE}")
        console.info(f"📝 Logs: {LOG_FILE}")
    else:
        console.error("Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
