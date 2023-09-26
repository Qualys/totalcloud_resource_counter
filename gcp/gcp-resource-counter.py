import logging
import os
import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import concurrent.futures

# Configure logging to save logs to a file
log_file = 'gcp_resource_counter.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)


# Function to list projects under a folder and its child folders recursively
def list_projects_recursive(service, folder_id):
    projects = []
    request = service.projects().list(parent=f"folders/{folder_id}")
    while request is not None:
        try:
            response = request.execute()
            for project in response.get('projects', []):
                projects.append(project)
            request = service.projects().list_next(previous_request=request, previous_response=response)
        except Exception as e:
            error_message = f"An error occurred while listing projects in folder {folder_id}: {str(e)}"
            logging.error(error_message)
            break
    # List child folders and their projects recursively
    child_folders = service.folders().list(parent=f"folders/{folder_id}").execute().get('folders', [])
    for child_folder in child_folders:
        projects += list_projects_recursive(service, child_folder['name'].split('/')[-1])
    return projects

# Function to list projects directly under an organization
def list_projects_under_organization(service, organization_id):
    projects = []
    request = service.projects().list(parent=f"organizations/{organization_id}")
    while request is not None:
        try:
            response = request.execute()
            for project in response.get('projects', []):
                projects.append(project)
            request = service.projects().list_next(previous_request=request, previous_response=response)
        except Exception as e:
            error_message = f"An error occurred while listing projects under organization {organization_id}: {str(e)}"
            logging.error(error_message)
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
        logging.error(error_message)
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
        logging.error(error_message)
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
        logging.error(error_message)
        return 0
# Function to count Artifacts Repositories and Docker Images in a project
def count_artifacts_and_docker_images(service, project_id):
    try:
        # Build the Artifact Registry client
        artifact_registry_service = build('artifactregistry', 'v1', credentials=credentials)

        # Fetch available locations for the project
        locations_response = artifact_registry_service.projects().locations().list(
            name=f'projects/{project_id}'
        ).execute()

        # Get the list of available locations
        locations = locations_response.get('locations', [])

        # Initialize counts
        total_repository_count = 0
        total_docker_image_count = 0

        # Iterate through locations and count repositories and Docker Images
        for location in locations:
            # List repositories in each location
            repositories_list = artifact_registry_service.projects().locations().repositories().list(
                parent=f'projects/{project_id}/locations/{location["locationId"]}'
            ).execute()

            # Get the count of repositories in each location
            repository_count = len(repositories_list.get('repositories', []))

            # List Docker images in each repository
            for repository in repositories_list.get('repositories', []):
                repository_name = repository['name']
                docker_images_list = artifact_registry_service.projects().locations().repositories().dockerImages().list(
                    parent=f'{repository_name}'
                ).execute()

                # Get the count of Docker images in each repository
                docker_image_count = len(docker_images_list.get('dockerImages', []))

                total_repository_count += 1
                total_docker_image_count += docker_image_count

        return total_repository_count, total_docker_image_count

    except HttpError as e:
        error_message = f"An error occurred while counting Artifacts Repositories and Docker Images in project {project_id}: {str(e)}"
        logging.error(error_message)
        return 0, 0

# Function to process resources for a single project
def process_resources(credentials, project_id):
    try:
        # Create the Resource Manager API client
        service = build('cloudresourcemanager', 'v3', credentials=credentials)

        info_message = f'Fetching Resources for Project ID: {project_id}'
        logging.info(info_message)

        # Count resources for the single project
        instances_count = count_running_compute_instances(service, project_id, credentials)
        functions_count = count_cloud_functions(service, project_id, credentials)
        gke_clusters_count = count_gke_clusters(service, project_id, credentials)
        repository_count, docker_image_count = count_artifacts_and_docker_images(service, project_id)

        result_message = f'Project: {project_id}, ' \
                         f'Running Compute Instances: {instances_count}, ' \
                         f'Cloud Functions: {functions_count}, ' \
                         f'GKE Clusters: {gke_clusters_count}, ' \
                         f'Artifacts Repositories: {repository_count}, ' \
                         f'Docker Images: {docker_image_count}'

        # Print the total counts for project
        logging.info(f"  Project ID {project_id} Resource Counts:")
        logging.info(f"  Running Compute Instances: {instances_count}")
        logging.info(f"  Cloud Functions: {functions_count}")
        logging.info(f"  GKE Clusters: {gke_clusters_count}")
        logging.info(f"  Docker Images: {docker_image_count}")

        # Create a CSV file to store the results
        csv_file = 'gcp_resource_counts.csv'
        script_directory = os.path.dirname(os.path.realpath(__file__))
        csv_file_path = os.path.join(script_directory, csv_file)
        # Check if the CSV file already exists, if not, write the header row
        if not os.path.isfile(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Write the header row
                csv_writer.writerow(
                    ['Project ID', 'Running Compute Instances', 'Cloud Functions', 'GKE Clusters', 'Artifacts Repositories',
                     'Docker Images'])

        # Append the project data to the CSV file
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the project data
            csv_writer.writerow([project_id, instances_count, functions_count, gke_clusters_count, repository_count,
                                 docker_image_count])
        # Display the path to the CSV file
        info_message = f'CSV file updated at: {csv_file_path}'
        logging.info(info_message)

    except Exception as e:
        error_message = f"An error occurred while processing project {project_id}: {str(e)}"
        logging.error(error_message)

# Function to run the code for an organization using multithreading
def run_for_organization(organization_id, service_account_key_file, credentials):
    # Create the Resource Manager API client
    service = build('cloudresourcemanager', 'v3', credentials=credentials)

    # List projects directly under the organization
    org_projects = list_projects_under_organization(service, organization_id)

    # List all folders under the organization
    folders = service.folders().list(parent=f"organizations/{organization_id}").execute().get('folders', [])

    # Create a list to store all project IDs
    all_project_ids = []

    # Add projects directly under the organization to the list
    for project in org_projects:
        all_project_ids.append(project['projectId'])

    # Iterate through the folders and list projects under each folder and its child folders
    for folder in folders:
        folder_id = folder['name'].split('/')[-1]  # Extract the folder ID from the folder name
        projects = list_projects_recursive(service, folder_id)
        for project in projects:
            all_project_ids.append(project['projectId'])

    # Convert the list to a set to ensure uniqueness and count the unique projects
    unique_project_ids = set(all_project_ids)

    # Display the unique project count
    info_message = f'Unique Project Count under Organization {organization_id}: {len(unique_project_ids)}'
    logging.info(info_message)

    # Create a thread pool to process resources concurrently for each project
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        # Submit tasks for processing resources for each project
        futures = {executor.submit(process_resources, credentials, project_id): project_id for project_id in unique_project_ids}

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            project_id = futures[future]
            try:
                future.result()  # Get the result of the completed task
            except Exception as e:
                error_message = f"An error occurred while processing project {project_id}: {str(e)}"
                logging.error(error_message)

    # Save the log file in the same directory as the script
    log_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_file)
    info_message = f'Log file saved at: {log_file_path}'
    logging.info(info_message)

# Function to run the code for a single project using provided credentials
def run_for_project(credentials):
    # Create the Resource Manager API client
    service = build('cloudresourcemanager', 'v3', credentials=credentials)

    # Fetch the project ID automatically from the provided credentials
    project_id = credentials.project_id

    info_message = f'Fetching Resources for Project ID: {project_id}'
    logging.info(info_message)

    # Count resources for the single project
    instances_count = count_running_compute_instances(service, project_id, credentials)
    functions_count = count_cloud_functions(service, project_id, credentials)
    gke_clusters_count = count_gke_clusters(service, project_id, credentials)
    repository_count, docker_image_count = count_artifacts_and_docker_images(service, project_id)

    result_message = f'Project: {project_id}, ' \
                     f'Running Compute Instances: {instances_count}, ' \
                     f'Cloud Functions: {functions_count}, ' \
                     f'GKE Clusters: {gke_clusters_count}, ' \
                     f'Artifacts Repositories: {repository_count}, ' \
                     f'Docker Images: {docker_image_count}'

    # Print the total counts for project
    logging.info(f"  Project ID {project_id} Resource Counts:")
    logging.info(f"  Running Compute Instances: {instances_count}")
    logging.info(f"  Cloud Functions: {functions_count}")
    logging.info(f"  GKE Clusters: {gke_clusters_count}")
    logging.info(f"  Docker Images: {docker_image_count}")

    # Create a CSV file to store the results
    csv_file = 'project_resource_counts.csv'
    script_directory = os.path.dirname(os.path.realpath(__file__))
    csv_file_path = os.path.join(script_directory, csv_file)
    # Check if the CSV file already exists, if not, write the header row
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row
            csv_writer.writerow(
                ['Project ID', 'Running Compute Instances', 'Cloud Functions', 'GKE Clusters', 'Artifacts Repositories',
                 'Docker Images'])

    # Append the project data to the CSV file
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the project data
        csv_writer.writerow([project_id, instances_count, functions_count, gke_clusters_count, repository_count,
                             docker_image_count])
    # Display the path to the CSV file
    info_message = f'CSV file updated at: {csv_file_path}'
    logging.info(info_message)

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
