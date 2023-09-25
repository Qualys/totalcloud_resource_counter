# Azure Resource Counter

**Author**: Yash Jhunjhunwala

This script allows you to count various types of Azure resources, such as virtual machines, web apps, container instances, AKS clusters, ACR repositories, and Azure Functions, within your Azure subscriptions and resource groups.

## Prerequisites

Before using this script, you must have the following prerequisites:

1. **Azure Subscription**: You need access to an Azure subscription to authenticate and access Azure resources.

2. **Azure Service Principal**: Create an Azure Service Principal with the appropriate permissions. You will need the following details:
   - **Client ID**
   - **Client Secret**
   - **Tenant ID**

3. **Python Environment**: Ensure you have Python installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

4. **Azure Python SDK**: Install the Azure Python SDKs using pip:
   ```bash
   pip install azure-mgmt-compute azure-mgmt-resource azure-identity azure-mgmt-web azure-mgmt-containerinstance azure-mgmt-containerservice azure-mgmt-containerregistry tqdm

## Authentication
You need to provide the Azure Service Principal's Client ID, Client Secret, and Tenant ID to authenticate the script with Azure.

## Usage
Follow these steps to use the script:

1. Clone this GitHub repository to your local machine:
```bash
git clone https://github.com/Qualys/totalcloud_resource_counter.git
```

2. Change your working directory to the project folder:
```bash
cd totalcloud_resource_counter/azure
```
3. Install the required Python packages using pip:
```bash
pip3 install -r requirements.txt

4. Run the script:
```bash
python3 azure-resource-counter.py
```
5. you will be prompted to enter the Azure Client ID, Client Secret, and Tenant ID. Provide these details to authenticate the script.

6. Select an option:
  1. Process all subscriptions with progress bar: Counts resources in all Azure subscriptions.
  2. Process a single subscription: Counts resources in a single Azure subscription by entering its ID.

7. The script will start counting resources and save the results in a CSV file named **azure_resource_counts.csv** .

## Permissions
To ensure that the script can count resources in your Azure subscription, grant the Service Principal the following permissions:

This allows read-only access to Azure resources at the Tenant Level.
- Custom Role with below permission
  - Microsoft.Management/managementGroups/read
  - Microsoft.Management/managementGroups/subscriptions/read permissions in you
- System define ReaderRole
You can assign this role using the Azure Portal, Azure CLI, or Azure PowerShell.
