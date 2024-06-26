import boto3
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import csv
import logging

# Configure logging to write to a file
log_file_name = "aws_resource_count.log"
logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(levelname)s: %(message)s')

# Define the CSV file name
csv_file_name = "aws_resource_counts.csv"

def get_management_account_id(management_session):
    """
    Get the AWS account ID of the management account in the organization.

    Args:
        management_session (boto3.Session): Session for the management account.

    Returns:
        str: AWS account ID of the management account.
    """
    org_client = management_session.client('organizations')
    try:
        response = org_client.describe_organization()
        return response['Organization']['MasterAccountId']
    except Exception as e:
        logging.error(f"Error getting management account ID: {e}")
        return None

def assume_role_and_get_session(account_id, role_name, session_name, management_session):
    """
    Assume a role in a member account and return a session.

    Args:
        account_id (str): AWS account ID.
        role_name (str): Name of the IAM role to assume.
        session_name (str): Name for the assumed session.
        management_session (boto3.Session): Session for the management account.

    Returns:
        boto3.Session: Session for the assumed role.
        Exception: Error encountered during the role assumption, if any.
    """
    sts_client = management_session.client('sts')
    role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'

    try:
        response = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name
        )

        # Create a session using temporary credentials
        member_session = boto3.Session(
            aws_access_key_id=response['Credentials']['AccessKeyId'],
            aws_secret_access_key=response['Credentials']['SecretAccessKey'],
            aws_session_token=response['Credentials']['SessionToken']
        )

        return member_session, None
    except Exception as e:
        return None, e

def count_resources_in_region(account_session, region_name):
    """
    Count resources concurrently in a specific region.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        dict: Resource counts in the region.
    """
    counts = {
        'running_ec2_instances': count_all_ec2_instances_in_region(account_session, region_name),
        'lambda_functions': count_lambda_functions_in_region(account_session, region_name),
        'ecs_fargate_tasks': count_ecs_fargate_tasks_in_region(account_session, region_name),
        'eks_instances': count_eks_instances_in_region(account_session, region_name),
        'ecr_repositories': count_ecr_repositories_in_region(account_session, region_name),
        'ecr_images': count_ecr_images_in_region(account_session, region_name),
        'eks_nodes': count_eks_nodes_in_region(account_session, region_name),  # Add EKS node counting
    }
    return counts

# Add a new function to count EKS nodes in a region
import boto3

def count_eks_nodes_in_region(account_session, region_name):
    """
    Count unique EKS nodes in a specific region, including nodes within nodegroups.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        int: Count of unique EKS nodes.
    """
    try:
        eks_client = account_session.client('eks', region_name=region_name)
        unique_nodes = set()

        # Use paginator to handle pagination for listing clusters
        cluster_paginator = eks_client.get_paginator('list_clusters')
        for cluster_page in cluster_paginator.paginate():
            for cluster_name in cluster_page.get('clusters', []):
                # List instances directly associated with the cluster's VPC
                vpc_id = eks_client.describe_cluster(name=cluster_name)['cluster']['resourcesVpcConfig']['vpcId']
                ec2_client = boto3.client('ec2', region_name=region_name)
                instance_response = ec2_client.describe_instances(Filters=[
                    {'Name': 'vpc-id', 'Values': [vpc_id]},
                    {'Name': 'tag-key', 'Values': ['kubernetes.io/cluster/' + cluster_name]}
                ])
                for reservation in instance_response['Reservations']:
                    for instance in reservation['Instances']:
                        if instance['State']['Name'] == 'running':
                            unique_nodes.add(instance['InstanceId'])  # Add instance ID to set of unique nodes

                # Use paginator to handle pagination for listing nodegroups
                nodegroup_paginator = eks_client.get_paginator('list_nodegroups')
                for nodegroup_page in nodegroup_paginator.paginate(clusterName=cluster_name):
                    for nodegroup_name in nodegroup_page.get('nodegroups', []):
                        try:
                            nodegroup_details = eks_client.describe_nodegroup(
                                clusterName=cluster_name,
                                nodegroupName=nodegroup_name
                            )
                            # Add instance IDs from nodegroup to set of unique nodes
                            for instance in nodegroup_details['nodegroup'].get('instances', []):
                                unique_nodes.add(instance['id'])
                        except KeyError as e:
                            print(f"KeyError occurred while processing nodegroup '{nodegroup_name}' in region {region_name}: {e}")
                        except Exception as e:
                            print(f"An error occurred while processing nodegroup '{nodegroup_name}' in region {region_name}: {e}")

        return len(unique_nodes)
    except Exception as e:
        print(f"An error occurred in region {region_name}: {e}")
        return 0

def count_all_ec2_instances_in_region(account_session, region_name):
    """
    Count all EC2 instances in a specific region.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        int: Count of all EC2 instances.
    """
    try:
        ec2_client = account_session.client('ec2', region_name=region_name)

        total_ec2_instance_count = 0
        paginator = ec2_client.get_paginator('describe_instances')

        # Paginate through DescribeInstances API call
        for page in paginator.paginate():
            for reservation in page['Reservations']:
                total_ec2_instance_count += len(reservation['Instances'])

        return total_ec2_instance_count

    except Exception as e:
        print(f"An error occurred in region {region_name}: {e}")
        return 0


def count_lambda_functions_in_region(account_session, region_name):
    """
    Count Lambda functions in a specific region.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        int: Count of Lambda functions.
    """
    try:
        lambda_client = account_session.client('lambda', region_name=region_name)
        response = lambda_client.list_functions()

        lambda_function_count = len(response.get('Functions', []))
        return lambda_function_count
    except Exception as e:
        return 0

def count_ecs_fargate_tasks_in_region(account_session, region_name):
    """
    Count ECS Fargate tasks in a specific region.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        int: Count of ECS Fargate tasks.
    """
    try:
        ecs_client = account_session.client('ecs', region_name=region_name)
        response = ecs_client.list_clusters()

        ecs_fargate_task_count = 0

        for cluster_arn in response.get('clusterArns', []):
            tasks_response = ecs_client.list_tasks(cluster=cluster_arn)
            ecs_fargate_task_count += len(tasks_response.get('taskArns', []))

        return ecs_fargate_task_count
    except Exception as e:
        return 0

def count_eks_instances_in_region(account_session, region_name):
    """
    Count EKS instances in a specific region.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        int: Count of EKS instances.
    """
    try:
        eks_client = account_session.client('eks', region_name=region_name)
        response = eks_client.list_clusters()

        eks_instance_count = len(response.get('clusters', []))
        return eks_instance_count
    except Exception as e:
        return 0


def count_ecr_repositories_in_region(account_session, region_name):
    """
    Count ECR repositories in a specific region with pagination support.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        int: Count of ECR repositories.
    """
    try:
        ecr_client = account_session.client('ecr', region_name=region_name)

        repository_count = 0
        paginator = ecr_client.get_paginator('describe_repositories')

        for page in paginator.paginate():
            repositories = page.get('repositories', [])
            repository_count += len(repositories)

        return repository_count
    except Exception as e:
        return 0


def count_ecr_images_in_region(account_session, region_name):
    """
    Count ECR images in a specific region.

    Args:
        account_session (boto3.Session): Session for the AWS account.
        region_name (str): AWS region name.

    Returns:
        int: Count of ECR images.
    """
    try:
        ecr_client = account_session.client('ecr', region_name=region_name)
        paginator = ecr_client.get_paginator('describe_repositories')

        ecr_image_count = 0

        for page in paginator.paginate():
            for repository in page.get('repositories', []):
                image_paginator = ecr_client.get_paginator('describe_images')
                for image_page in image_paginator.paginate(repositoryName=repository['repositoryName']):
                    ecr_image_count += len(image_page.get('imageDetails', []))

        return ecr_image_count
    except Exception as e:
        print(f"An error occurred in region {region_name}: {e}")
        return 0

def get_active_regions(account_session):
    """
    Get the list of active regions for an AWS account.

    Args:
        account_session (boto3.Session): Session for the AWS account.

    Returns:
        list: List of active AWS region names.
    """
    try:
        ec2_client = account_session.client('ec2', region_name='us-east-1')  # Use us-east-1 as a common region
        regions = [region['RegionName'] for region in ec2_client.describe_regions()['Regions']]
        return regions, None
    except Exception as e:
        return None, e

def count_resources(input_type, management_access_key, management_secret_key):
    """
    Count AWS resources either at the organization or account level.

    Args:
        input_type (str): Resource counting type ('org' or 'account').
        management_access_key (str): Access key for the management account.
        management_secret_key (str): Secret key for the management account.

    Returns:
        None
    """
    # Create a session for the management account
    management_session = boto3.Session(
        aws_access_key_id=management_access_key,
        aws_secret_access_key=management_secret_key
    )

    # Get the management account ID
    management_account_id = get_management_account_id(management_session)

    if not management_account_id:
        logging.error("Failed to retrieve the management account ID.")
        exit()

    # Create an org_client outside of the if block
    org_client = management_session.client('organizations')

    # Create a list to store the results
    results = []

    # Organization-level resource counting
    if input_type == 'org':
        # Gather resources for the management account using provided credentials
        # Get the list of active regions for the management account
        active_regions, regions_error = get_active_regions(management_session)

        if regions_error:
            logging.error(f"Error getting active regions for the management account: {regions_error}")
            exit()

        # Initialize counts to zero for the management account
        total_running_ec2_instances = 0
        total_lambda_function_count = 0
        total_ecs_fargate_task_count = 0
        total_eks_instance_count = 0
        total_ecr_repository_count = 0
        total_ecr_image_count = 0
        total_eks_node_count = 0  # Initialize EKS node count

        # Iterate over active regions and count resources concurrently
        with ThreadPoolExecutor(max_workers=30) as executor:
            resource_counts = list(executor.map(
                count_resources_in_region,
                [management_session] * len(active_regions),
                active_regions
            ))

        # Aggregate resource counts from different regions
        for counts in resource_counts:
            total_running_ec2_instances += counts['running_ec2_instances']
            total_lambda_function_count += counts['lambda_functions']
            total_ecs_fargate_task_count += counts['ecs_fargate_tasks']
            total_eks_instance_count += counts['eks_instances']
            total_ecr_repository_count += counts['ecr_repositories']
            total_ecr_image_count += counts['ecr_images']
            total_eks_node_count += counts['eks_nodes']  # Add EKS node count

        # Print the total counts for the management account
        logging.info(f"  Management Account {management_account_id} Resource Counts:")
        logging.info(f"  Total Running EC2 Instances: {total_running_ec2_instances}")
        logging.info(f"  Total Lambda Functions: {total_lambda_function_count}")
        logging.info(f"  Total ECS Fargate Tasks: {total_ecs_fargate_task_count}")
        logging.info(f"  Total EKS Instances: {total_eks_instance_count}")
        logging.info(f"  Total ECR Repositories: {total_ecr_repository_count}")
        logging.info(f"  Total ECR Images: {total_ecr_image_count}")
        logging.info(f"  Total EKS Nodes: {total_eks_node_count}\n")  # Log EKS node count

        # Append the results to the list
        results.append([
            management_account_id,
            total_running_ec2_instances,
            total_lambda_function_count,
            total_ecs_fargate_task_count,
            total_eks_instance_count,
            total_ecr_repository_count,
            total_ecr_image_count,
            total_eks_node_count  # Add EKS node count
        ])

        # Append the results to the CSV file immediately
        with open(csv_file_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the data rows to the CSV file
            csv_writer.writerows(results)

        # Clear results for the next member account
        results.clear()

        # Gather resources for member accounts
        # List all member account IDs within the organization with progress bar
        member_account_ids = []

        paginator = org_client.get_paginator('list_accounts')
        for page in tqdm(paginator.paginate(), desc="Fetching Member Accounts"):
            for account in page['Accounts']:
                member_account_ids.append(account['Id'])

        # Loop through each member account
        for member_account_id in tqdm(member_account_ids, desc="Processing Member Accounts"):
            # Create a new session for each member account by assuming the role
            member_account_session, session_error = assume_role_and_get_session(
                member_account_id,
                'OrganizationAccountAccessRole',
                'CountResources',
                management_session
            )

            if session_error:
                logging.error(f"Error creating session for account {member_account_id}: {session_error}")
                continue

            # Get the list of active regions for the member account
            active_regions, regions_error = get_active_regions(member_account_session)

            if regions_error:
                logging.error(f"Error getting active regions for account {member_account_id}: {regions_error}")
                continue

            # Initialize counts to zero for each member account
            total_running_ec2_instances = 0
            total_lambda_function_count = 0
            total_ecs_fargate_task_count = 0
            total_eks_instance_count = 0
            total_ecr_repository_count = 0
            total_ecr_image_count = 0
            total_eks_node_count = 0  # Initialize EKS node count

            # Iterate over active regions and count resources concurrently
            with ThreadPoolExecutor(max_workers=30) as executor:
                resource_counts = list(executor.map(
                    count_resources_in_region,
                    [member_account_session] * len(active_regions),
                    active_regions
                ))

            # Aggregate resource counts from different regions
            for counts in resource_counts:
                total_running_ec2_instances += counts['running_ec2_instances']
                total_lambda_function_count += counts['lambda_functions']
                total_ecs_fargate_task_count += counts['ecs_fargate_tasks']
                total_eks_instance_count += counts['eks_instances']
                total_ecr_repository_count += counts['ecr_repositories']
                total_ecr_image_count += counts['ecr_images']
                total_eks_node_count += counts['eks_nodes']  # Add EKS node count

            # Print the total counts for all regions in the member account
            logging.info(f"Member Account {member_account_id} Resource Counts:")
            logging.info(f"  Total Running EC2 Instances: {total_running_ec2_instances}")
            logging.info(f"  Total Lambda Functions: {total_lambda_function_count}")
            logging.info(f"  Total ECS Fargate Tasks: {total_ecs_fargate_task_count}")
            logging.info(f"  Total EKS Instances: {total_eks_instance_count}")
            logging.info(f"  Total ECR Repositories: {total_ecr_repository_count}")
            logging.info(f"  Total ECR Images: {total_ecr_image_count}")
            logging.info(f"  Total EKS Nodes: {total_eks_node_count}\n")  # Log EKS node count

            # Append the results to the list
            results.append([
                member_account_id,
                total_running_ec2_instances,
                total_lambda_function_count,
                total_ecs_fargate_task_count,
                total_eks_instance_count,
                total_ecr_repository_count,
                total_ecr_image_count,
                total_eks_node_count  # Add EKS node count
            ])

            # Append the results to the CSV file immediately
            with open(csv_file_name, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                # Write the data rows to the CSV file
                csv_writer.writerows(results)

            # Clear results for the next member account
            results.clear()

        logging.info(f"Results added to {csv_file_name}")

    # Account-level resource counting
    else:
        # Get the account ID associated with the provided access keys
        sts_client = management_session.client('sts')

        try:
            response = sts_client.get_caller_identity()
            account_id = response['Account']
        except Exception as e:
            logging.error(f"Error getting account ID: {e}")
            exit()

        # Get the list of active regions for the account
        active_regions, regions_error = get_active_regions(management_session)

        if regions_error:
            logging.error(f"Error getting active regions for the account: {regions_error}")
            exit()

        # Initialize counts to zero for the account
        total_running_ec2_instances = 0
        total_lambda_function_count = 0
        total_ecs_fargate_task_count = 0
        total_eks_instance_count = 0
        total_ecr_repository_count = 0
        total_ecr_image_count = 0
        total_eks_node_count = 0  # Initialize EKS node count

        # Iterate over active regions and count resources concurrently
        with ThreadPoolExecutor(max_workers=30) as executor:
            resource_counts = list(executor.map(
                count_resources_in_region,
                [management_session] * len(active_regions),
                active_regions
            ))

        # Aggregate resource counts from different regions
        for counts in resource_counts:
            total_running_ec2_instances += counts['running_ec2_instances']
            total_lambda_function_count += counts['lambda_functions']
            total_ecs_fargate_task_count += counts['ecs_fargate_tasks']
            total_eks_instance_count += counts['eks_instances']
            total_ecr_repository_count += counts['ecr_repositories']
            total_ecr_image_count += counts['ecr_images']
            total_eks_node_count += counts['eks_nodes']  # Add EKS node count

        # Print the total counts for all regions in the account
        logging.info(f"Account {account_id} Resource Counts:")
        logging.info(f"  Total Running EC2 Instances: {total_running_ec2_instances}")
        logging.info(f"  Total Lambda Functions: {total_lambda_function_count}")
        logging.info(f"  Total ECS Fargate Tasks: {total_ecs_fargate_task_count}")
        logging.info(f"  Total EKS Instances: {total_eks_instance_count}")
        logging.info(f"  Total ECR Repositories: {total_ecr_repository_count}")
        logging.info(f"  Total ECR Images: {total_ecr_image_count}")
        logging.info(f"  Total EKS Nodes: {total_eks_node_count}")  # Log EKS node count

        # Append the results to the CSV file
        with open(csv_file_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the data rows to the CSV file
            csv_writer.writerow([
                account_id,
                total_running_ec2_instances,
                total_lambda_function_count,
                total_ecs_fargate_task_count,
                total_eks_instance_count,
                total_ecr_repository_count,
                total_ecr_image_count,
                total_eks_node_count  # Add EKS node count
            ])

        logging.info(f"Results added to {csv_file_name}")

if __name__ == "__main__":
    # Input type: 'org' for organization-level resource counting or 'account' for account-level counting
    input_type = input("Enter 'org' for organization-level resource counting or 'account' for account-level counting: ")

    if input_type not in ('org', 'account'):
        logging.error("Invalid input. Please enter 'org' or 'account'.")
        exit()

    # Management account access key and secret key
    management_access_key = input("Enter the access key for the account: ")
    management_secret_key = input("Enter the secret key for the account: ")

    # Initialize the CSV file with headers
    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row to the CSV file
        csv_writer.writerow([
            'Account ID',
            'Running EC2 Instances',
            'Lambda Functions',
            'ECS Fargate Tasks',
            'EKS Instances',
            'ECR Repositories',
            'ECR Images',
            'EKS Nodes'  # Add EKS node count
        ])

    # Count AWS resources based on the input type
    count_resources(input_type, management_access_key, management_secret_key)
