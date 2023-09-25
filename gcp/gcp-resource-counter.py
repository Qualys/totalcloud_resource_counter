import logging
import os
import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm
import concurrent.futures


# Configure logging to save logs to a file
log_file = 'gcp_resource_counter.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# Function to list projects under a folder and its child folders recursively
def list_projects_recursive(service, folder_id, pbar):
    projects = []
    request = service.projects().list(parent=f"folders/{folder_id}")
    while request is not None:
        try:
            response = request.execute()
            for project in response.get('projects', []):
                projects.append(project)
                pbar.update(1)
            request = service.projects().list_next(previous_request=request, previous_response=response)
        except Exception as e:
            error_message = f"An error occurred while listing projects in folder {folder_id}: {str(e)}"
            logger.error(error_message)
            break

    # List child folders and their projects recursively
    child_folders = service.folders().list(parent=f"folders/{folder_id}").execute().get('folders', [])
    for child_folder in child_folders:
        projects += list_projects_recursive(service, child_folder['name'].split('/')[-1], pbar)

    return projects


# Function to list projects directly under an organization
def list_projects_under_organization(service, organization_id, pbar):
    projects = []
    request = service.projects().list(parent=f"organizations/{organization_id}")
    while request is not None:
        try:
            response = request.execute()
            for project in response.get('projects', []):
                projects.append(project)
                pbar.update(1)
            request = service.projects().list_next(previous_request=request, previous_response=response)
        except Exception as e:
            error_message = f"An error occurred while listing projects under organization {organization_id}: {str(e)}"
            logger.error(error_message)
            break

    return projects


# Function to count running compute instances in a project
def count_running_compute_instances(service, project_id, credentials):
    try:
        compute_service = build('compute', 'v1', credentials=credentials)

        # List running compute instances, paginating through results
        instances_count = 0
        request = compute_service.instances().aggregatedList(project=project_id)
        while request is not None:
            response = request.execute()
            for zone, instances_data in response.get('items', {}).items():
                for instance in instances_data.get('instances', []):
                    instances_count += 1
            if 'nextPageToken' in response:
                request = compute_service.instances().aggregatedList(
                    project=project_id, pageToken=response['nextPageToken'])
            else:
                break

        return instances_count
    except HttpError as e:
        error_message = f"An error occurred while counting running compute instances in project {project_id}: {str(e)}"
        logger.error(error_message)
        return 0


# Function to count Cloud Functions in a project
def count_cloud_functions(service, project_id, credentials):
    try:
        functions_service = build('cloudfunctions', 'v1', credentials=credentials)

        # List Cloud Functions in all regions, paginating through results
        functions_count = 0
        request = functions_service.projects().locations().functions().list(
            parent=f"projects/{project_id}/locations/-")
        while request is not None:
            response = request.execute()
            functions_count += len(response.get('functions', []))
            if 'nextPageToken' in response:
                request = functions_service.projects().locations().functions().list(
                    parent=f"projects/{project_id}/locations/-",
                    pageToken=response['nextPageToken'])
            else:
                break

        return functions_count
    except HttpError as e:
        error_message = f"An error occurred while counting Cloud Functions in project {project_id}: {str(e)}"
        logger.error(error_message)
        return 0


# Function to list Artifact Registry repositories in a project
def list_artifact_registry_repositories(service, project_id):
    try:
        artifact_registry_service = build('artifactregistry', 'v1', credentials=credentials)

        # List Artifact Registry repositories in the project
        repositories_list = artifact_registry_service.projects().locations().repositories().list(
            parent=f"projects/{project_id}/locations/-").execute()

        repositories_count = []

        if 'repositories' in repositories_list:
            for repository in repositories_list['repositories']:
                repositories_count.append(repository['name'])

        return repositories_count
    except HttpError as e:
        error_message = f"An error occurred while listing Artifact Registry repositories in project {project_id}: {str(e)}"
        logger.error(error_message)
        return []


# Function to count Artifact Registry Docker images in a repository
def count_artifact_registry_images(service, project_id, location, repository_name, credentials):
    try:
        artifact_registry_service = build('artifactregistry', 'v1', credentials=credentials)

        # List Artifact Registry Docker images in the repository
        images_list = artifact_registry_service.projects().locations().repositories().dockerImages().list(
            parent=f"projects/{project_id}/locations/{location}/repositories/{repository_name}").execute()

        images_count = 0

        if 'dockerImages' in images_list:
            images_count = len(images_list['dockerImages'])

        return images_count
    except HttpError as e:
        error_message = f"An error occurred while counting Artifact Registry Docker images in repository {repository_name}: {str(e)}"
        logger.error(error_message)
        return 0


# Function to count GKE clusters in a project
def count_gke_clusters(service, project_id, credentials):
    try:
        container_service = build('container', 'v1', credentials=credentials)

        # List GKE clusters in the project
        clusters_list = container_service.projects().locations().clusters().list(
            parent=f"projects/{project_id}/locations/-").execute()

        clusters_count = 0

        if 'clusters' in clusters_list:
            clusters_count = len(clusters_list['clusters'])

        return clusters_count
    except HttpError as e:
        error_message = f"An error occurred while counting GKE clusters in project {project_id}: {str(e)}"
        logger.error(error_message)
        return 0


# Function to count resources for a single project
def count_resources_for_project(service, project_id, credentials):
    instances_count = count_running_compute_instances(service, project_id, credentials)
    functions_count = count_cloud_functions(service, project_id, credentials)
    gke_clusters_count = count_gke_clusters(service, project_id, credentials)
    artifact_registry_repositories = list_artifact_registry_repositories(service, project_id)

    # Initialize the progress bar for counting Artifact Registry images
    with tqdm(total=len(artifact_registry_repositories), dynamic_ncols=True,
              desc=f"Counting Artifact Registry Images in Project {project_id}") as artifact_registry_pbar:
        artifact_registry_images_counts = []
        for repo in artifact_registry_repositories:
            location = "us-central1"  # Change to the appropriate location
            images_count = count_artifact_registry_images(service, project_id, location, repo, credentials,
                                                          artifact_registry_pbar)
            artifact_registry_images_counts.append(images_count)

    artifact_registry_images_count = sum(artifact_registry_images_counts)

    return {
        'Project ID': project_id,
        'Running Compute Instances': instances_count,
        'Cloud Functions': functions_count,
        'GKE Clusters': gke_clusters_count,
        'Artifact Registry Repositories': len(artifact_registry_repositories),
        'Artifact Registry Images': artifact_registry_images_count,
    }



# Function to run the code for an organization
def run_for_organization(organization_id, service_account_key_file, credentials):
    # Create the Resource Manager API client
    service = build('cloudresourcemanager', 'v3', credentials=credentials)

    # List projects directly under the organization
    org_projects = list_projects_under_organization(service, organization_id, pbar=tqdm(total=0, dynamic_ncols=True,
                                                                                        desc="Listing Projects in Org"))

    # List all folders under the organization
    folders = service.folders().list(parent=f"organizations/{organization_id}").execute().get('folders', [])

    # Create a list to store all project IDs
    all_project_ids = []

    # Add projects directly under the organization to the list
    for project in org_projects:
        all_project_ids.append(project['projectId'])

    # Initialize the progress bar for listing projects under folders
    with tqdm(total=len(folders), dynamic_ncols=True, desc="Listing Projects in Folders") as pbar:
        # Iterate through the folders and list projects under each folder and its child folders
        for folder in folders:
            folder_id = folder['name'].split('/')[-1]  # Extract the folder ID from the folder name
            projects = list_projects_recursive(service, folder_id, pbar)
            for project in projects:
                all_project_ids.append(project['projectId'])

    # Convert the list to a set to ensure uniqueness and count the unique projects
    unique_project_ids = set(all_project_ids)

    # Display the unique project count
    info_message = f'Unique Project Count under Organization {organization_id}: {len(unique_project_ids)}'
    logger.info(info_message)

    # Create a list to store the results for each project
    project_results = []

    # Initialize the progress bar for counting resources
    with tqdm(total=len(unique_project_ids) * 5, dynamic_ncols=True, desc="Counting Resources") as pbar:
        # Create a list of projects to process
        projects_to_process = list(unique_project_ids)

        # Initialize a ThreadPoolExecutor to parallelize resource counting
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use ThreadPoolExecutor to parallelize resource counting
            for result in executor.map(lambda project_id: count_resources_for_project(service, project_id, credentials), projects_to_process):
                project_results.append(result)
                pbar.update(5)

    # Display the count of resources for each unique project
    info_message = "\nResource Counts for each Unique Project:"
    logger.info(info_message)
    for result in project_results:
        result_message = f'Project: {result["Project ID"]}, ' \
                         f'Running Compute Instances: {result["Running Compute Instances"]}, ' \
                         f'Cloud Functions: {result["Cloud Functions"]}, ' \
                         f'GKE Clusters: {result["GKE Clusters"]}, ' \
                         f'Artifact Registry Repositories: {result["Artifact Registry Repositories"]}, ' \
                         f'Artifact Registry Images: {result["Artifact Registry Images"]}'
        logger.info(result_message)

    # Save the results to a CSV file (in append mode for single project)
    csv_file = 'organization_resource_counts.csv'
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, csv_file)

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(['Project ID', 'Running Compute Instances', 'Cloud Functions', 'GKE Clusters',
                             'Artifact Registry Repositories', 'Artifact Registry Images'])

        # Write the project data
        for result in project_results:
            csv_writer.writerow([result['Project ID'], result['Running Compute Instances'], result['Cloud Functions'],
                                 result['GKE Clusters'], result['Artifact Registry Repositories'],
                                 result['Artifact Registry Images']])

    # Display the path to the CSV file
    info_message = f'CSV file saved at: {csv_file_path}'
    logger.info(info_message)

    # Save the log file in the same directory as the script
    log_file_path = os.path.join(script_directory, log_file)
    info_message = f'Log file saved at: {log_file_path}'
    logger.info(info_message)


# Function to run the code for a single project using provided credentials
def run_for_project(credentials):
    # Create the Resource Manager API client
    service = build('cloudresourcemanager', 'v3', credentials=credentials)

    # Fetch the project ID automatically from the provided credentials
    project_id = credentials.project_id

    # Count resources for the single project
    instances_count = count_running_compute_instances(service, project_id, credentials)
    functions_count = count_cloud_functions(service, project_id, credentials)
    gke_clusters_count = count_gke_clusters(service, project_id, credentials)
    artifact_registry_repositories = list_artifact_registry_repositories(service, project_id)

    # Initialize the progress bar for counting Artifact Registry images
    with tqdm(total=len(artifact_registry_repositories), dynamic_ncols=True,
              desc="Counting Artifact Registry Images") as artifact_registry_pbar:
        artifact_registry_images_counts = []
        for repo in artifact_registry_repositories:
            location = "us-central1"  # Change to the appropriate location
            images_count = count_artifact_registry_images(service, project_id, location, repo, credentials)
            artifact_registry_images_counts.append(images_count)
            artifact_registry_pbar.update(1)

    artifact_registry_images_count = sum(artifact_registry_images_counts)

    # Display the count of resources for the project
    info_message = "\nResource Counts for the Project:"
    logger.info(info_message)
    result_message = f'Project: {project_id}, ' \
                     f'Running Compute Instances: {instances_count}, ' \
                     f'Cloud Functions: {functions_count}, ' \
                     f'GKE Clusters: {gke_clusters_count}, ' \
                     f'Artifact Registry Repositories: {len(artifact_registry_repositories)}, ' \
                     f'Artifact Registry Images: {artifact_registry_images_count}'
    logger.info(result_message)

    # CSV file path with header row
    csv_file = 'project_resource_counts.csv'
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, csv_file)

    # Check if the CSV file already exists, if not, write the header row
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row
            csv_writer.writerow(['Project ID', 'Running Compute Instances', 'Cloud Functions', 'GKE Clusters',
                                 'Artifact Registry Repositories', 'Artifact Registry Images'])

    # Append the project data to the CSV file
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write the project data
        csv_writer.writerow([project_id, instances_count, functions_count,
                             gke_clusters_count, len(artifact_registry_repositories),
                             artifact_registry_images_count])

    # Display the path to the CSV file
    info_message = f'CSV file updated at: {csv_file_path}'
    logger.info(info_message)


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run for an entire organization")
    print("2. Run for a single project")

    option = input("Enter your choice (1 or 2): ")

    if option == "1":
        # Run for an entire organization
        organization_id = input("Enter your organization ID: ")
        service_account_key_file = input("Enter the path to your service account key JSON file: ")
        credentials = service_account.Credentials.from_service_account_file(
            service_account_key_file, scopes=['https://www.googleapis.com/auth/cloud-platform'])
        run_for_organization(organization_id, service_account_key_file, credentials)
    elif option == "2":
        # Run for a single project using the provided credentials
        credentials_path = input("Enter the path to your service account key JSON file: ")
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=['https://www.googleapis.com/auth/cloud-platform'])
        run_for_project(credentials)
    else:
        print("Invalid option. Please choose 1 or 2.")
