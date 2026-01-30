# Annotation

## Installation

Run 
```bash
./realtimevenv/bin/python -m pip install fr24sdk
./realtimevenv/bin/python -m pip install folium
```

## Authentication

### macOS/Linux
'export FR24_API_TOKEN="your_actual_token_here"'

### Windows (PowerShell)
'$Env:FR24_API_TOKEN="your_actual_token_here"'

## Annotation 

Change the following parameters according to your setup: MICROPHONE_LOC, basefolder, dt_local_start, dt_local_end

Run the following to obtain the ground truth file:
```bash
python -m dataset.annotate
```


