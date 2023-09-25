# GCP Resource Counter

**Author**: Yash Jhunjhunwala

This Python script is a command-line tool for counting various Google Cloud Platform (GCP) resources within a GCP organization or a single GCP project. It uses the Google Cloud Client Libraries for Python to interact with GCP services and APIs.

## Features

- Lists GCP projects under an organization and its child folders recursively.
- Counts running compute instances, Cloud Functions, GKE clusters, Artifact Registry repositories, and Artifact Registry Docker images in each project.
- Saves the resource counts to CSV files for further analysis.
- Provides progress bars during resource counting.
- Logs informational and error messages.

## Prerequisites

Before running the script, ensure you have the following:

- Python 3.x installed on your system.
- Required Python libraries installed (install them using `pip`):
  - `google-auth`
  - `google-auth-oauthlib`
  - `google-auth-httplib2`
  - `google-api-python-client`
  - `tqdm`
 
### Google Cloud Platform (GCP) Account

- You must have a Google Cloud Platform (GCP) account.
- The account should have the necessary permissions to access and manage GCP resources.

### Service Account Key JSON File

- You'll need a service account key JSON file with the appropriate permissions.
- Create a service account in your GCP project or organization.
- Assign the necessary roles and permissions to the service account, such as:
  - `roles/compute.viewer` for counting compute instances.
  - `roles/cloudfunctions.viewer` for counting Cloud Functions.
  - `roles/container.viewer` for counting GKE clusters.
  - `roles/artifactregistry.reader` for listing Artifact Registry repositories and images.
  - `roles/browser` for viewing organizations and folders in GCP.
  - `roles/resourcemanager.folderViewer` for viewing folders and projects within the organization.
  - Ensure the service account has at least read access to the GCP resources you want to count.

## Getting Started

1. Clone this repository to your local machine:

   ```bash
  https://github.com/yjhunjhun/resource-counter.git

2. Navigate to the project directory:
   ```bash
   cd resource-counter/gcp

3. Navigate to the project directory:
   ```bash
   pip3 install -r requirements.txt

4. To count AWS resources, run the script and follow the prompts.
   ```shell
   python3 gcp-resource-counter.py

5. Choose an option to run the script:
  Option 1: Run for an entire organization
    - Enter your organization ID when prompted.
    - Provide the path to your service account key JSON file when prompted.
  Option 2: Run for a single project
    - Enter the path to your service account key JSON file for the specific GCP project.
    - Follow the prompts and let the script run.
  
## Output
The script generates CSV files:
 - organization_resource_counts.csv (when running for an organization)
 - project_resource_counts.csv (when running for a single project)

## Logging
Log messages are saved to gcp_resource_counter.log.

### Contributing
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Create a pull request with a clear description of your changes.
