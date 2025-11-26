import pandas as pd
import numpy as np
from pyulog.core import ULog
import matplotlib.pyplot as plt

def load_ulg(ulog_file_name, timestamp_field="timestamp", msg_filter=None, disable_str_exceptions=False, 
             verbose=False, relative_timestamps=True, skip_first_n=20, fields_to_extract=None, startTime=None):
    """
    Load ULog file and return numpy arrays with timestamps for each requested field.
    
    Args:
        ulog_file_name: Path to ULog file
        timestamp_field: Name of timestamp field to use
        msg_filter: Message filter for ULog
        disable_str_exceptions: Disable string exceptions
        verbose: Print verbose output
        relative_timestamps: Make timestamps relative to start
        skip_first_n: Number of samples to skip if timestamps not monotonic
        fields_to_extract: List of [section_name, field_name] pairs to extract.
                          If None, extracts all fields.
                          Format: [['section_name1', 'field_name1'], ['section_name2', 'field_name2'], ...]
        startTime: Start time in seconds. Data before this time will be cropped, and timestamps will be
                   shifted so that startTime becomes zero. If None, no cropping/shifting is performed.
    
    Returns:
        Dictionary with field identifiers as keys. Each value is a dict with:
            - 'timestamp': numpy array of timestamps (in seconds, relative if relative_timestamps=True, 
                           shifted by startTime if startTime is provided)
            - 'data': numpy array of data values (NaN values removed)
        Format: {'section_name_field_name': {'timestamp': np.array, 'data': np.array}, ...}
    """
    ulog = ULog(ulog_file_name, msg_filter, disable_str_exceptions)
    data = ulog.data_list
    result = {}  # Dictionary to store results: {field_id: {'timestamp': array, 'data': array}}
    
    # Convert fields_to_extract to a set of tuples for faster lookup
    fields_set = None
    if fields_to_extract is not None:
        fields_set = set((section, field) for section, field in fields_to_extract)
        if verbose:
            print(f"Extracting {len(fields_set)} specific fields: {fields_to_extract}")
    
    # List of possible timestamp field names to try (in order of preference)
    timestamp_field_candidates = [timestamp_field, "timestamp_sample", "timestamp"]
    # Remove duplicates while preserving order
    timestamp_field_candidates = list(dict.fromkeys(timestamp_field_candidates))
    
    # Track message type occurrences to handle duplicate message types
    message_type_count = {}
    
    for d in data:
        # Skip this section if fields_to_extract is specified and this section is not in the list
        if fields_set is not None:
            # Check if any field from this section is requested
            section_fields = [f.field_name for f in d.field_data]
            section_requested = any((d.name, field) in fields_set for field in section_fields)
            if not section_requested:
                if verbose:
                    print(f"Skipping section {d.name} (not in fields_to_extract)")
                continue
        data_keys = [f.field_name for f in d.field_data]

        assert(len(data_keys) > 0)
        assert(all([len(d.data[key]) == len(d.data[data_keys[0]]) for key in data_keys]))
        print(f"{d.name} has {len(d.data[data_keys[0]])} entries") if verbose else None
        
        # Try to find a valid timestamp field
        found_timestamp_field = None
        for candidate in timestamp_field_candidates:
            if candidate in data_keys:
                found_timestamp_field = candidate
                break
        
        if found_timestamp_field is None:
            print(f"Warning: No timestamp field found in {d.name}. Tried: {timestamp_field_candidates}")
            continue
        
        data_keys.remove(found_timestamp_field)
        
        # Filter data_keys to only include requested fields if fields_to_extract is specified
        if fields_set is not None:
            data_keys = [key for key in data_keys if (d.name, key) in fields_set]
            if len(data_keys) == 0:
                if verbose:
                    print(f"No requested fields found in section {d.name}, skipping")
                continue
        
        timestamps = np.array(d.data[found_timestamp_field])
        
        # Check if timestamps are monotonic (strictly increasing)
        is_monotonic = len(timestamps) > 0 and np.all(timestamps[1:] > timestamps[:-1])
        
        # If not monotonic, skip first N samples
        skip_n = 0
        if not is_monotonic:
            skip_n = min(skip_first_n, len(timestamps))
            if skip_n > 0:
                timestamps = timestamps[skip_n:]
                # Also skip first N for all data fields
                for key in data_keys:
                    d.data[key] = d.data[key][skip_n:]
                if verbose:
                    print(f"Warning: Timestamps in {d.name} are not monotonic. Skipping first {skip_n} samples.")
        
        # Track occurrences of this message type to handle duplicates
        if d.name not in message_type_count:
            message_type_count[d.name] = 0
        else:
            message_type_count[d.name] += 1
        
        # Add suffix if this message type has appeared before to ensure unique field names
        message_suffix = f"_{message_type_count[d.name]}" if message_type_count[d.name] > 0 else ""
        
        # Convert timestamps to seconds
        timestamps_sec = timestamps / 1e6
        
        # Crop data from startTime if specified
        if startTime is not None:
            crop_mask = timestamps_sec >= startTime
            if np.any(crop_mask):
                timestamps_sec = timestamps_sec[crop_mask]
                # Crop corresponding data fields
                for key in data_keys:
                    d.data[key] = d.data[key][crop_mask]
                if verbose:
                    cropped_count = np.sum(~crop_mask)
                    if cropped_count > 0:
                        print(f"Section {d.name}: Cropped {cropped_count} samples before startTime={startTime}s")
            else:
                if verbose:
                    print(f"Warning: Section {d.name} has no data after startTime={startTime}s, skipping")
                continue
        
        # Shift timestamps by startTime if specified (so startTime becomes zero)
        if startTime is not None and len(timestamps_sec) > 0:
            timestamps_sec = timestamps_sec - startTime
        
        # Make timestamps relative if requested (only if startTime was not used, as it already shifts)
        if relative_timestamps and startTime is None and len(timestamps_sec) > 0:
            timestamps_sec = timestamps_sec - timestamps_sec[0]
        
        # Process each field and store with its timestamp
        for key in data_keys:
            # Create unique field identifier
            field_id = f"{d.name}{message_suffix}_{key}"
            
            # Get data as numpy array
            data_array = np.array(d.data[key], dtype=np.float64)
            
            # Remove NaN values and corresponding timestamps
            valid_mask = ~np.isnan(data_array)
            if np.any(valid_mask):
                # Only keep valid (non-NaN) data points
                clean_timestamps = timestamps_sec[valid_mask]
                clean_data = data_array[valid_mask]
                
                # Store in result dictionary
                result[field_id] = {
                    'timestamp': clean_timestamps,
                    'data': clean_data
                }
                
                if verbose:
                    removed_count = np.sum(~valid_mask)
                    if removed_count > 0:
                        print(f"Field {field_id}: Removed {removed_count} NaN values, kept {len(clean_data)} valid samples")
            else:
                if verbose:
                    print(f"Warning: Field {field_id} contains only NaN values, skipping")
    
    return result



def timeframe(df, time_start, time_end):
    df_timeframe = df[(df.index > time_start) & (df.index < time_end)].copy()
    return df_timeframe