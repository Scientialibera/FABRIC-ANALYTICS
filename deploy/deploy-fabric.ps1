param(
    [string]$ConfigPath = "deploy/deploy.config.toml"
)

$ErrorActionPreference = "Stop"

function Write-Step { param([string]$Message) Write-Host "[deploy-fabric] $Message" }

function Get-Config {
    param([string]$Path)
    if (-not (Test-Path $Path)) { throw "Config file not found: $Path" }
    $json = python -c "import json, pathlib, tomllib; p=pathlib.Path(r'$Path'); print(json.dumps(tomllib.loads(p.read_text(encoding='utf-8'))))"
    if ($LASTEXITCODE -ne 0) { throw "Failed to parse config file: $Path" }
    return $json | ConvertFrom-Json
}

function Select-Value {
    param([string]$Configured, [string]$Default)
    if ([string]::IsNullOrWhiteSpace($Configured)) { return $Default }
    return $Configured
}

function Get-FabricToken {
    return az account get-access-token --resource "https://api.fabric.microsoft.com" --query accessToken -o tsv
}

function Invoke-FabricApi {
    param(
        [string]$Method,
        [string]$Uri,
        [object]$Body = $null
    )
    $token = Get-FabricToken
    $headers = @{ Authorization = "Bearer $token" }
    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $headers
    }
    $headers["Content-Type"] = "application/json"
    $jsonBody = $Body | ConvertTo-Json -Depth 50
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $headers -Body $jsonBody
}

function Get-FabricItems {
    param([string]$WorkspaceId, [string]$Type = "")
    $base = "https://api.fabric.microsoft.com/v1/workspaces/$WorkspaceId/items"
    $uri = if ([string]::IsNullOrWhiteSpace($Type)) { $base } else { "$base?type=$Type" }
    $response = Invoke-FabricApi -Method "GET" -Uri $uri
    if ($response -and $response.value) { return @($response.value) }
    return @()
}

function Ensure-FabricFolder {
    param(
        [string]$WorkspaceId,
        [string]$DisplayName,
        [string]$ParentFolderId = ""
    )
    $foldersResponse = Invoke-FabricApi -Method "GET" -Uri "https://api.fabric.microsoft.com/v1/workspaces/$WorkspaceId/folders"
    $folders = if ($foldersResponse -and $foldersResponse.value) { @($foldersResponse.value) } else { @() }
    $existing = $folders | Where-Object {
        $_.displayName -eq $DisplayName -and (
            ([string]::IsNullOrWhiteSpace($ParentFolderId) -and [string]::IsNullOrWhiteSpace($_.parentFolderId)) -or
            ($_.parentFolderId -eq $ParentFolderId)
        )
    } | Select-Object -First 1
    if ($existing) { return $existing.id }

    $body = @{ displayName = $DisplayName }
    if (-not [string]::IsNullOrWhiteSpace($ParentFolderId)) { $body.parentFolderId = $ParentFolderId }
    $created = Invoke-FabricApi -Method "POST" -Uri "https://api.fabric.microsoft.com/v1/workspaces/$WorkspaceId/folders" -Body $body
    return $created.id
}

function Ensure-FabricLakehouse {
    param(
        [string]$WorkspaceId,
        [string]$LakehouseId,
        [string]$LakehouseName,
        [string]$FolderId
    )
    if (-not [string]::IsNullOrWhiteSpace($LakehouseId)) { return $LakehouseId }
    $existing = Get-FabricItems -WorkspaceId $WorkspaceId -Type "Lakehouse" | Where-Object {
        $_.displayName -eq $LakehouseName
    } | Select-Object -First 1
    if ($existing) { return $existing.id }

    $body = @{ displayName = $LakehouseName; type = "Lakehouse" }
    if (-not [string]::IsNullOrWhiteSpace($FolderId)) { $body.folderId = $FolderId }
    $created = Invoke-FabricApi -Method "POST" -Uri "https://api.fabric.microsoft.com/v1/workspaces/$WorkspaceId/items" -Body $body
    return $created.id
}

function Ensure-FabricNotebook {
    param(
        [string]$WorkspaceId,
        [string]$DisplayName,
        [string]$FolderId,
        [string]$SourceFilePath
    )
    if (-not (Test-Path $SourceFilePath)) { throw "Notebook source file not found: $SourceFilePath" }
    $source = Get-Content -Path $SourceFilePath -Raw -Encoding UTF8
    $payloadBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($source))
    $definition = @{
        format = "fabricGitSource"
        parts = @(
            @{
                path = "notebook-content.py"
                payload = $payloadBase64
                payloadType = "InlineBase64"
            }
        )
    }

    $existing = Get-FabricItems -WorkspaceId $WorkspaceId -Type "Notebook" | Where-Object {
        $_.displayName -eq $DisplayName
    } | Select-Object -First 1

    if (-not $existing) {
        $body = @{
            displayName = $DisplayName
            type = "Notebook"
            folderId = $FolderId
            definition = $definition
        }
        $created = Invoke-FabricApi -Method "POST" -Uri "https://api.fabric.microsoft.com/v1/workspaces/$WorkspaceId/items" -Body $body
        Write-Step "Created notebook '$DisplayName'."
        return $created.id
    }

    $updateBody = @{ definition = $definition }
    Invoke-FabricApi -Method "POST" -Uri "https://api.fabric.microsoft.com/v1/workspaces/$WorkspaceId/items/$($existing.id)/updateDefinition?updateMetadata=true" -Body $updateBody | Out-Null
    Write-Step "Updated notebook '$DisplayName'."
    return $existing.id
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
$config = Get-Config -Path $ConfigPath
$workspaceId = $config.fabric.workspace_id
if ([string]::IsNullOrWhiteSpace($workspaceId)) { throw "fabric.workspace_id is required." }

Write-Step "Ensuring Fabric folder tree."
$rootFolderId = Ensure-FabricFolder -WorkspaceId $workspaceId -DisplayName "notebooks"
$mainFolderId = Ensure-FabricFolder -WorkspaceId $workspaceId -DisplayName "main" -ParentFolderId $rootFolderId
$modulesFolderId = Ensure-FabricFolder -WorkspaceId $workspaceId -DisplayName "modules" -ParentFolderId $rootFolderId

$lhSection = $config.lakehouses

Write-Step "Ensuring lakehouses."
$landingId = Ensure-FabricLakehouse -WorkspaceId $workspaceId -LakehouseId $lhSection.landing_id -LakehouseName $lhSection.landing_name -FolderId ""
$bronzeId  = Ensure-FabricLakehouse -WorkspaceId $workspaceId -LakehouseId $lhSection.bronze_id  -LakehouseName $lhSection.bronze_name  -FolderId ""
$silverId  = Ensure-FabricLakehouse -WorkspaceId $workspaceId -LakehouseId $lhSection.silver_id  -LakehouseName $lhSection.silver_name  -FolderId ""
$goldId    = Ensure-FabricLakehouse -WorkspaceId $workspaceId -LakehouseId $lhSection.gold_id    -LakehouseName $lhSection.gold_name    -FolderId ""

Write-Step "Pushing module notebooks."
$modulesBase = "deploy/assets/notebooks/modules"
foreach ($file in Get-ChildItem -Path $modulesBase -Filter "*.py") {
    $displayName = $file.BaseName
    Ensure-FabricNotebook -WorkspaceId $workspaceId -DisplayName $displayName -FolderId $modulesFolderId -SourceFilePath $file.FullName | Out-Null
}

Write-Step "Pushing main notebooks."
$mainBase = "deploy/assets/notebooks/main"
foreach ($file in Get-ChildItem -Path $mainBase -Filter "*.py" | Sort-Object Name) {
    $displayName = $file.BaseName
    Ensure-FabricNotebook -WorkspaceId $workspaceId -DisplayName $displayName -FolderId $mainFolderId -SourceFilePath $file.FullName | Out-Null
}

$result = @{
    workspace_id = $workspaceId
    landing_lakehouse_id = $landingId
    bronze_lakehouse_id = $bronzeId
    silver_lakehouse_id = $silverId
    gold_lakehouse_id = $goldId
}

Write-Step "Deployment complete."
Write-Output ($result | ConvertTo-Json -Compress)
