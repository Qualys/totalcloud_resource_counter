# Azure Resource Counter

**Author**: Yash Jhunjhunwala

This script allows you to count various types of Azure resources, such as virtual machines, web apps, container instances, AKS clusters, ACR repositories, and Azure Functions, within your Azure subscriptions and resource groups.

## Features

- **Resource Counting**: Easily count and aggregate the number of various Azure resources within a subscription.
- **Granular Resource Types**: Count specific resource types, including Virtual Machines, Web Apps, Container Instances, AKS Clusters, ACR Registries, ACR Images, and Azure Functions.
- **Subscription Selection**: Choose to count resources for all subscriptions associated with your Azure account or focus on a specific subscription.
- **CSV Export**: Export the resource counts to a CSV file for further analysis or reporting.
- **Detailed Logging**: View detailed logs for each resource count operation and any encountered errors.
- **Interactive User Interface**: An interactive command-line interface guides you through the process.


## Prerequisites

Before using this script, you need to have the following prerequisites in place:

- Azure Client ID
- Azure Client Secret
- Azure Tenant ID
- Python 3.x installed
- Azure SDK for Python installed (`azure-mgmt-compute`, `azure-mgmt-resource`, `azure-mgmt-subscription`, `azure-mgmt-web`, `azure-mgmt-containerinstance`, `azure-mgmt-containerservice`, `azure-mgmt-containerregistry`)

## Permissions
To ensure that the script can count resources in your Azure subscription, grant the Service Principal the following permissions:

This allows read-only access to Azure resources at the Tenant Level.
- Custom Role with below permission
  - Microsoft.Management/managementGroups/read
  - Microsoft.Management/managementGroups/subscriptions/read permissions in you
- System define ReaderRole
You can assign this role using the Azure Portal, Azure CLI, or Azure PowerShell.

## Getting Started
Follow these steps to use the script:

1. Clone this GitHub repository to your local machine:
   ```shell
   git clone https://github.com/Qualys/totalcloud_resource_counter.git

2. Change your working directory to the project folder:
   ```shell
   cd totalcloud_resource_counter/azure

3. Install dependencies in a virtual environment (recommended):
   ```shell
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt

4. Run the script:
   ```shell
   python3 azure-resource-counter.py

5. You will be prompted to enter the Azure Client ID, Client Secret, and Tenant ID. Provide these details to authenticate the script.

6. Select an option:
   - Process all subscriptions with progress bar: Counts resources in all Azure subscriptions.
   - Process a single subscription: Counts resources in a single Azure subscription by entering its ID.

## Output
The script will generate a CSV file named ***azure_resource_counts.csv*** that contains resource counts for each Azure subscription.

## Logging

Logging is an essential part of this script, providing detailed information about its execution and any encountered errors. Here's how logging is configured and used:

- **Logging Configuration**: Logging is configured using the Python `logging` module. Logs are saved to separate log files:
  - `azure_resource_counts.log`: Contains general information and execution logs.
  - `azure_resource_counts_error.log`: Contains error messages and stack traces for encountered errors.

- **Log Levels**: Different log levels are used to categorize log messages:
  - `INFO`: Informational messages about the progress of the script.
  - `ERROR`: Error messages for exceptions and issues during execution.

- **Log Files**: Two log files are created in the script's directory: `azure_resource_counts.log` for general logs and `azure_resource_counts_error.log` for error logs.
