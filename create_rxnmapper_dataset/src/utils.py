
def find_input_type(file_path):
    if file_path.endswith('.csv'):
        return 'csv'
    elif file_path.endswith('.json'):
        return 'json'
    elif file_path.endswith('.txt'):
        return 'txt'
    
    elif file_path.endswith('.txt_log_probs'):
        return 'txt_log_probs'
    
    elif file_path.endswith('.pkl'):
        return 'pkl'
    else:
        raise ValueError('File type not supported')