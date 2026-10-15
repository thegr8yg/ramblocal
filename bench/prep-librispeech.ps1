# Downloads LibriSpeech test-clean, converts FLAC to 16 kHz mono WAV, writes bench/data/test-clean/manifest.jsonl.
# Requires ffmpeg on PATH.
$ErrorActionPreference = 'Stop'
$data = Join-Path $PSScriptRoot 'data'
New-Item -ItemType Directory -Force $data | Out-Null
$tgz = Join-Path $data 'test-clean.tar.gz'
if (-not (Test-Path $tgz)) {
  Invoke-WebRequest 'https://www.openslr.org/resources/12/test-clean.tar.gz' -OutFile $tgz
}
if (-not (Test-Path (Join-Path $data 'LibriSpeech'))) { tar -xzf $tgz -C $data }

$out = Join-Path $data 'test-clean'
New-Item -ItemType Directory -Force $out | Out-Null
$lines = [System.Collections.Generic.List[string]]::new()
Get-ChildItem (Join-Path $data 'LibriSpeech/test-clean') -Recurse -Filter *.trans.txt | ForEach-Object {
  $dir = $_.DirectoryName
  Get-Content $_.FullName | ForEach-Object {
    $id, $text = $_ -split ' ', 2
    $wav = Join-Path $out "$id.wav"
    if (-not (Test-Path $wav)) {
      ffmpeg -loglevel error -y -i (Join-Path $dir "$id.flac") -ar 16000 -ac 1 -c:a pcm_s16le $wav
    }
    $lines.Add((@{ audio = "$id.wav"; text = $text } | ConvertTo-Json -Compress))
  }
}
Set-Content (Join-Path $out 'manifest.jsonl') $lines
"wrote $($lines.Count) items"
