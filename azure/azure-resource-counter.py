import logging
import csv
from azure.identity import ClientSecretCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.subscription import SubscriptionClient
from azure.mgmt.web import WebSiteManagementClient
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerservice import ContainerServiceClient
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from tqdm import tqdm
import threading
import os

# Constants
CSV_FILE_PATH = 'azure_resource_counts.csv'
ERROR_LOG_FILE_PATH = 'azure_resource_counts_error.log'

# Configure logging
logging.basicConfig(filename='azure_resource_counts.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
error_logger = logging.getLogger('azure_errors')
error_logger.setLevel(logging.ERROR)

# Set the log level for the "azure" logger to WARNING to exclude request logs
azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)

# Authenticate and return an Azure client credential.
def authenticate_client(client_id, client_secret, tenant_id):
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    return credential

# List Azure subscriptions using the provided credential
def list_subscriptions(credential):
    subscription_client = SubscriptionClient(credential)
    return list(subscription_client.subscriptions.list())

# Function to count resources in a resource group
def count_resources_in_resource_group(credential, subscription_id, resource_group, result_dict):
    counts = {
        "Virtual Machines": count_virtual_machines(credential, subscription_id, resource_group),
        "Web Apps": count_web_apps(credential, subscription_id, resource_group),
        "Container Instances": count_container_instances(credential, subscription_id, resource_group),
        "AKS Clusters": count_aks_clusters(credential, subscription_id, resource_group),
        "ACR Registries": count_acr_registries(credential, subscription_id, resource_group),
        "ACR Images": count_acr_images(credential, subscription_id, resource_group),
        "Azure Functions": count_azure_functions(credential, subscription_id, resource_group),
    }

    result_dict[resource_group] = counts

# Function to count virtual machines in a resource group
def count_virtual_machines(credential, subscription_id, resource_group):
    compute_client = ComputeManagementClient(credential, subscription_id)
    total_vm_count = 0

    try:
        vm_paged = compute_client.virtual_machines.list(resource_group)
        for _ in vm_paged:
            total_vm_count += 1

        return total_vm_count
    except Exception as e:
        logging.error(f"Error counting VMs in {resource_group}: {str(e)}")
        return 0

# Function to count web apps in a resource group
def count_web_apps(credential, subscription_id, resource_group):
    website_client = WebSiteManagementClient(credential, subscription_id)
    total_web_app_count = 0

    try:
        web_apps = website_client.web_apps.list_by_resource_group(resource_group)
        for _ in web_apps:
            total_web_app_count += 1

        return total_web_app_count
    except Exception as e:
        logging.error(f"Error counting Web Apps in {resource_group}: {str(e)}")
        return 0

# Function to count container instances in a resource group
def count_container_instances(credential, subscription_id, resource_group):
    container_instance_client = ContainerInstanceManagementClient(credential, subscription_id)
    total_aci_count = 0

    try:
        aci_paged = container_instance_client.container_groups.list_by_resource_group(resource_group)
        for _ in aci_paged:
            total_aci_count += 1

        return total_aci_count
    except Exception as e:
        logging.error(f"Error counting ACIs in {resource_group}: {str(e)}")
        return 0

# Function to count AKS clusters in a resource group
def count_aks_clusters(credential, subscription_id, resource_group):
    container_service_client = ContainerServiceClient(credential, subscription_id)
    total_aks_count = 0

    try:
        aks_paged = container_service_client.managed_clusters.list_by_resource_group(resource_group)
        for _ in aks_paged:
            total_aks_count += 1

        return total_aks_count
    except Exception as e:
        logging.error(f"Error counting AKS clusters in {resource_group}: {str(e)}")
        return 0

# Function to count ACR registries in a resource group
def count_acr_registries(credential, subscription_id, resource_group):
    container_registry_client = ContainerRegistryManagementClient(credential, subscription_id)
    total_acr_registry_count = 0

    try:
        acr_registries = container_registry_client.registries.list_by_resource_group(resource_group)
        for _ in acr_registries:
            total_acr_registry_count += 1

        return total_acr_registry_count
    except Exception as e:
        logging.error(f"Error counting ACR registries in {resource_group}: {str(e)}")
        return 0



# Function to count ACR images in a resource group
def count_acr_images(credential, subscription_id, resource_group):
    container_registry_client = ContainerRegistryManagementClient(credential, subscription_id)
    total_acr_image_count = 0

    try:
        acr_repos = container_registry_client.registries.list_by_resource_group(resource_group)
        for repo in acr_repos:
            if hasattr(repo, 'registry_name'):
                registry_name = repo.registry_name

                # List the repositories in the registry
                repositories = container_registry_client.repositories.list(resource_group, registry_name)

                for repository in repositories:
                    # List the manifests within each repository
                    manifests = container_registry_client.manifests.list(resource_group, registry_name, repository)
                    total_acr_image_count += len(list(manifests))

        return total_acr_image_count
    except Exception as e:
        error_logger.error(f"Error counting ACR images in {resource_group}: {str(e)}")
        return 0

# Function to count Azure Functions in a resource group
def count_azure_functions(credential, subscription_id, resource_group):
    web_client = WebSiteManagementClient(credential, subscription_id)
    total_azure_functions = 0

    try:
        web_apps = web_client.web_apps.list_by_resource_group(resource_group)
        for app in web_apps:
            if app.kind and "functionapp" in app.kind.lower():
                total_azure_functions += 1

        return total_azure_functions
    except Exception as e:
        logging.error(f"Error counting Azure Functions in {resource_group}: {str(e)}")
        return 0




# Function to process a subscription and append counts to CSV
def process_subscription(credential, subscription, csv_writer):
    subscription_id = subscription.subscription_id
    logging.info(f"Processing Subscription ID: {subscription_id}")

    # Retrieve the list of resource groups in the subscription
    resource_groups = ResourceManagementClient(credential, subscription_id).resource_groups.list()

    # Create a dictionary to store counts for each resource group
    result_dict = {}

    # Create threads for counting resources in each resource group
    threads = []
    for resource_group in resource_groups:
        thread = threading.Thread(target=count_resources_in_resource_group,
                                  args=(credential, subscription_id, resource_group.name, result_dict))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Sum counts across resource groups
    total_counts = {
        "Virtual Machines": 0,
        "Web Apps": 0,
        "Container Instances": 0,
        "AKS Clusters": 0,
        "ACR Registries": 0,
        "ACR Images": 0,
        "Azure Functions": 0,
    }

    for counts in result_dict.values():
        for resource_type, count in counts.items():
            total_counts[resource_type] += count

    # Log total counts for each resource type
    for resource_type, count in total_counts.items():
        logging.info(f"Total {resource_type}: {count}")

    # Log a message if no resources are found for any type
    for resource_type, count in total_counts.items():
        if count == 0:
            logging.info(f"No {resource_type} found in Subscription ID {subscription_id}")
    # Append counts to the CSV for this subscription
    csv_writer.writerow([subscription_id] + list(total_counts.values()))
    logging.info(f"Counts for Subscription ID {subscription_id} saved in CSV")
    logging.info("-" * 50)




# Main function to process subscriptions
def main():
    client_id = input("Enter Azure Client ID: ")
    client_secret = input("Enter Azure Client Secret: ")
    tenant_id = input("Enter Azure Tenant ID: ")

    # Authenticate and create Azure credential
    credential = authenticate_client(client_id, client_secret, tenant_id)

    # List Azure subscriptions
    subscriptions = list_subscriptions(credential)
    logging.info(f"Total Subscriptions: {len(subscriptions)}")
    logging.info("-" * 50)

    # Create or append to the CSV file
    csv_exists = os.path.exists(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header row if the CSV file is newly created
        if not csv_exists:
            csv_writer.writerow(['Subscription ID', 'Virtual Machines', 'Web Apps', 'Container Instances', 'AKS Clusters',
                                 'ACR Registries', 'ACR Images', 'Azure Functions'])

        # Prompt the user to select the option
        print("Select an option:")
        print("1. Process all subscriptions")
        print("2. Process a single subscription")
        option = input("Enter your choice (1/2): ")

        if option == "1":
            # Process all subscriptions
            pbar = tqdm(subscriptions, desc="Processing Subscriptions", unit="subscription")
            for subscription in pbar:
                process_subscription(credential, subscription, csv_writer)
        elif option == "2":
            # Process a single subscription
            subscription_id = input("Enter the Subscription ID to process: ")
            subscription = next((sub for sub in subscriptions if sub.subscription_id == subscription_id), None)
            if subscription:
                process_subscription(credential, subscription, csv_writer)
            else:
                print("Subscription not found.")
        else:
            print("Invalid option. Please select 1 or 2.")

if __name__ == "__main__":
    main()
