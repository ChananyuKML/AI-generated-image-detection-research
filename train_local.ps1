<#
.SYNOPSIS
Train a local binary detector using the project's virtual environment.
#>
[CmdletBinding()]
param(
    [ValidateSet('learned_filter_cnn', 'fixed_filter_cnn', 'dual_cnn_128', 'fixed_filter_kcnet', 'finetuned_cnn_kcnet')]
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
    [string]$EncoderCheckpoint,
    [ValidateRange(0, 2147483647)]
    [int]$Patience,
    [ValidateScript({ $_ -ge 0 -and -not [double]::IsNaN($_) -and -not [double]::IsInfinity($_) })]
    [double]$MinDelta,
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
if ($EncoderCheckpoint) { $trainingArguments += @('--encoder-checkpoint', $EncoderCheckpoint) }
if ($PSBoundParameters.ContainsKey('Patience')) { $trainingArguments += @('--patience', [string]$Patience) }
if ($PSBoundParameters.ContainsKey('MinDelta')) { $trainingArguments += @('--min-delta', $MinDelta.ToString('G', [System.Globalization.CultureInfo]::InvariantCulture)) }
if ($CheckOnly) { $trainingArguments += '--check-only' }
& $projectPython @trainingArguments
exit $LASTEXITCODE
