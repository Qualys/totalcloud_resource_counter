# GCP Resource Counter

**Author**: Yash Jhunjhunwala

This Python script is a command-line tool for counting various Google Cloud Platform (GCP) resources within a GCP organization or a single GCP project. It uses the Google Cloud Client Libraries for Python to interact with GCP services and APIs.

## Features

- **Count GCP Resources**: This script allows you to count various Google Cloud Platform (GCP) resources, including Compute Engine instances, Cloud Functions, Google Kubernetes Engine (GKE) clusters, Artifacts Repositories, and Docker Images.
- **Run for Projects or Organizations**: You can choose to run the script for a single GCP project or for an entire organization. It provides flexibility for resource counting.
- **Concurrent Processing**: The script uses multithreading to process resources concurrently, which can significantly reduce the execution time, especially for organizations with many projects.
- **Detailed Output**: The script provides detailed information about the resources being counted, including the count of running compute instances, Cloud Functions, GKE clusters, Artifacts Repositories, and Docker Images.
- **Output in CSV**: The script generates CSV files containing resource counts, making it easy to analyze and share the results.

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
- **Enable APIs**: Make sure that the following GCP APIs are enabled for your project:
   - Compute Engine API
   - Cloud Functions API
   - Kubernetes Engine API
   - Artifact Registry API

## Getting Started

1. Clone this repository to your local machine:
   ```sh
   git clone https://github.com/Qualys/totalcloud_resource_counter.git
  
2. Navigate to the project directory:
   ```shell
   cd resource-counter/gcp

3. Install dependencies in a virtual environment (recommended):
   ```shell
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt

4. To count AWS resources, run the script and follow the prompts.
   ```shell
   python3 gcp-resource-counter.py

5. Choose an option to run the script:
   
- Option 1: Run for an entire organization
    - Enter your organization ID when prompted.
    - Provide the path to your service account key JSON file when prompted.

- Option 2: Run for a single project
    - Enter the path to your service account key JSON file for the specific GCP project.
    - Follow the prompts and let the script run.
  
## Output
The script will generate output in the following ways:

***Console Output:*** You will see detailed information about the resources being counted displayed in your terminal as the script runs.

***CSV Files:*** The script will create CSV files containing resource counts. The file names are as follows:

- When running the script for a single project: ***project_resource_counts.csv***
- When running the script for an organization: ***gcp_resource_counts.csv***
These CSV files will be located in the project directory.


## Logging

Logging is an essential part of this script, providing detailed information about its execution and any encountered errors. Here's how logging is configured and used:

- **Logging Configuration**: Logging is configured using the Python `logging` module. Logs are saved to separate log file:
  - `gcp_resource_counter.log`: Contains general information and execution logs.

- **Log Levels**: Different log levels are used to categorize log messages:
  - `INFO`: Informational messages about the progress of the script.
  - `ERROR`: Error messages for exceptions and issues during execution.
