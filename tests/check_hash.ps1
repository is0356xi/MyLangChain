Param(
    [Parameter(mandatory=$true)][String]$file_path
)

$hash_dict = @{'ggml-gpt4all-j-v1.3-groovy.bin'='81a09a0ddf89690372fc296ff7f625af'}
echo $hash_dict

Get-FileHash -Algorithm MD5 $file_path | Select-Object -Property Hash, Path