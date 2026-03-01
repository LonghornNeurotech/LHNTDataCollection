# 1. Record a test session
# 2. Stop and save
# 3. Verify with pyxdf:

import pyxdf

try:
    streams, header = pyxdf.load_xdf('your_file.xdf')
    print(f"✅ SUCCESS! Loaded {len(streams)} streams")
    
    for i, stream in enumerate(streams):
        print(f"\nStream {i}:")
        print(f"  Name: {stream['info']['name'][0]}")
        print(f"  Type: {stream['info']['type'][0]}")
        print(f"  Samples: {len(stream['time_stamps'])}")
        print(f"  Duration: {stream['time_stamps'][-1] - stream['time_stamps'][0]:.2f}s")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()