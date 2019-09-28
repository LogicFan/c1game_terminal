$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "$scriptPath\random.agent.py"

py -3 $algoPath
