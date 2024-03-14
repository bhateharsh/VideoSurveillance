import re

string = "DifferentLabel99_x264_456"

# Regular expression to extract format  *_x264
pattern = r"^(\w+)_x264"  # Simplified pattern

match = re.search(pattern, string)

if match:
  # Extract the desired part (label name)
  extracted_string = match.group(1)
  print(extracted_string)
else:
  print("No match found")
















