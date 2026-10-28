param(
    [ValidateSet('list', 'download', 'verify')]
    [string]$Mode = 'list',
    [string]$Output,
    [string[]]$Generators = @('midjourney', 'sdv4', 'sdv5', 'adm', 'glide', 'wukong', 'vqdm', 'biggan'),
    [ValidateSet('google-drive', 'huggingface')]
    [string]$Source = 'google-drive',
    [string[]]$Files,
    [int]$Retries = 3,
    [switch]$TestArchives,
    [string]$SevenZip = '7z'
)

$ErrorActionPreference = 'Stop'
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw 'Project .venv is missing. Run: python scripts/run.py setup --compute auto'
}
if ($Mode -ne 'list' -and [string]::IsNullOrWhiteSpace($Output)) {
    throw '-Output must name the archive directory on your data drive.'
}
$DownloadArguments = @((Join-Path $PSScriptRoot 'tools\download_genimage.py'))
switch ($Mode) {
    'list' { $DownloadArguments += '--list' }
    'download' { $DownloadArguments += '--download' }
    'verify' { $DownloadArguments += '--verify-only' }
}
if ($Output) { $DownloadArguments += @('--output', $Output) }
$DownloadArguments += @('--retries', [string]$Retries, '--generators') + $Generators
$DownloadArguments += @('--source', $Source)
if ($Files) { $DownloadArguments += @('--files') + $Files }
if ($TestArchives) { $DownloadArguments += @('--test-archives', '--seven-zip', $SevenZip) }
& $ProjectPython @DownloadArguments
exit $LASTEXITCODE
