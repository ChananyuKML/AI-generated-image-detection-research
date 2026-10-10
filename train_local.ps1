<#
.SYNOPSIS
Train a local binary detector using the project's virtual environment.
#>
[CmdletBinding()]
param(
    [ValidateSet('learned_filter_cnn', 'fixed_filter_cnn', 'dual_cnn_128')]
    [string]$Network = 'learned_filter_cnn',
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')]
    [string]$Dataset = 'sdv4',
    [string]$DataRoot = 'dataset/genimage',
    [int]$Epochs = 100,
    [int]$BatchSize = 32,
    [double]$LR = 1e-4,
    [int]$NumWorkers = 0,
    [int]$Seed = 42,
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'cuda',
    [string]$RunDir,
    [string]$Resume,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython -PathType Leaf)) {
    throw 'Run: python scripts/run.py setup --compute cu128 (or cpu) first.'
}
$trainingArguments = @(
    (Join-Path $PSScriptRoot 'train.py'),
    '--model', $Network, '--dataset', $Dataset, '--data-root', $DataRoot,
    '--epochs', [string]$Epochs, '--batch-size', [string]$BatchSize,
    '--lr', $LR.ToString('G', [System.Globalization.CultureInfo]::InvariantCulture),
    '--workers', [string]$NumWorkers, '--seed', [string]$Seed, '--device', $Device
)
if ($RunDir) { $trainingArguments += @('--run-dir', $RunDir) }
if ($Resume) { $trainingArguments += @('--resume', $Resume) }
if ($CheckOnly) { $trainingArguments += '--check-only' }
& $projectPython @trainingArguments
exit $LASTEXITCODE
