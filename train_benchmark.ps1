<#
.SYNOPSIS
Train one of the prepared proof-of-concept datasets with the local CUDA model.
.EXAMPLE
.\train_benchmark.ps1 -Dataset progan_poc
.EXAMPLE
.\train_benchmark.ps1 -Dataset cifake_poc -Network fixed_filter_cnn
.EXAMPLE
.\train_benchmark.ps1 -Dataset tiny_sdv5 -CheckOnly
#>
[CmdletBinding()]
param(
    [ValidateSet('cifake', 'cifake_poc', 'progan', 'progan_poc',
                 'tiny_genimage', 'tiny_genimage_poc', 'tiny_adm', 'tiny_biggan',
                 'tiny_glide', 'tiny_midjourney', 'tiny_sdv5', 'tiny_vqdm', 'tiny_wukong')]
    [string]$Dataset = 'progan_poc',

    [ValidateSet('learned_filter_cnn', 'fixed_filter_cnn', 'dual_cnn_128', 'fixed_filter_kcnet', 'finetuned_cnn_kcnet')]
    [string]$Network = 'learned_filter_cnn',

    [ValidateRange(1, 2147483647)]
    [int]$Epochs = 20,

    [ValidateRange(1, 2147483647)]
    [int]$BatchSize = 32,

    [ValidateScript({ $_ -gt 0 -and -not [double]::IsNaN($_) -and -not [double]::IsInfinity($_) })]
    [double]$LR = 1e-4,

    [ValidateRange(0, 2147483647)]
    [int]$NumWorkers = 0,

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 42,

    [string]$EncoderCheckpoint,

    [ValidateRange(0, 2147483647)]
    [int]$Patience,

    [ValidateScript({ $_ -ge 0 -and -not [double]::IsNaN($_) -and -not [double]::IsInfinity($_) })]
    [double]$MinDelta,

    [switch]$CheckOnly
)

$taskLauncher = Join-Path $PSScriptRoot 'train_local.ps1'
$taskParameters = @{
    Dataset = $Dataset
    Network = $Network
    DataRoot = (Join-Path $PSScriptRoot 'dataset\prepared')
    Epochs = $Epochs
    BatchSize = $BatchSize
    LR = $LR
    NumWorkers = $NumWorkers
    Seed = $Seed
}
if ($CheckOnly) { $taskParameters.CheckOnly = $true }
if ($EncoderCheckpoint) { $taskParameters.EncoderCheckpoint = $EncoderCheckpoint }
if ($PSBoundParameters.ContainsKey('Patience')) { $taskParameters.Patience = $Patience }
if ($PSBoundParameters.ContainsKey('MinDelta')) { $taskParameters.MinDelta = $MinDelta }
& $taskLauncher @taskParameters
exit $LASTEXITCODE
