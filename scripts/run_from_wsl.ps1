param(
    [string]$Distro = "Ubuntu-22.04",
    [string]$Script = "scripts/check_system.sh",
    [Alias("Args")]
    [string[]]$ScriptArgs = @()
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param([string]$WindowsPath)

    $resolved = (Resolve-Path $WindowsPath).Path
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $matches[1].ToLowerInvariant()
        $rest = $matches[2] -replace "\\", "/"
        return "/mnt/$drive/$rest"
    }

    throw "Only drive-letter Windows paths are supported: $resolved"
}

function Quote-Bash {
    param([string]$Value)

    if ($Value.Contains("'")) {
        throw "Arguments containing single quotes are not supported by this helper: $Value"
    }
    return "'$Value'"
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$WslRepoRoot = Convert-ToWslPath $RepoRoot

if ($Script.StartsWith("/")) {
    $WslScript = $Script
} else {
    $WslScript = "$WslRepoRoot/$($Script -replace "\\", "/")"
}

$QuotedArgs = @()
foreach ($Arg in $ScriptArgs) {
    $QuotedArgs += Quote-Bash $Arg
}

$Command = "cd $(Quote-Bash $WslRepoRoot) && bash $(Quote-Bash $WslScript)"
if ($QuotedArgs.Count -gt 0) {
    $Command += " " + ($QuotedArgs -join " ")
}

Write-Host "[icg] wsl -d $Distro -- bash -lc $Command"
wsl -d $Distro -- bash -lc $Command
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    exit $ExitCode
}
