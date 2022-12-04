import pickle

import pandas as pd

service_target_encoder = pickle.load(open("service_target_encoder.pkl", "rb"))
flag_target_encoder = pickle.load(open("flag_target_encoder.pkl", "rb"))
protocol_type_enc = pickle.load(open("protocol_type_enc.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
req_columns = pickle.load(open("req_columns.pkl", "rb"))

headers = ['duration', 'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp', 'service', 'flag',
               'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
               'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
               'srv_count', 'serror_rate',
               'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
               'srv_diff_host_rate', 'dst_host_count',
               'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate',
               'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
               'attack', 'level']

header = ['duration','protocol_type_icmp','protocol_type_tcp','protocol_type_udp','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
            'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
           'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
           'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
           'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
           'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class']

def dp_preprocessing(data):
    test_data = data
    test_data["service"] = service_target_encoder.transform(test_data["service"])
    test_data['service'] = test_data['service'].apply(lambda x: '%.6f' % float(x)).astype(
        float)  # Converting the precision value after decimal point to 6
    test_data["flag"] = flag_target_encoder.transform(test_data["flag"])
    test_data['flag'] = test_data['flag'].apply(lambda x: '%.6f' % float(x)).astype(
        float)  # Converting the precision value after decimal point to 6
    enc_protocol_df = pd.DataFrame(protocol_type_enc.transform(test_data[['protocol_type']]).toarray())
    enc_protocol_df.columns = ["protocol_type_icmp", "protocol_type_tcp", "protocol_type_udp"]
    test_data["protocol_type_icmp"] = enc_protocol_df["protocol_type_icmp"]
    test_data["protocol_type_tcp"] = enc_protocol_df["protocol_type_tcp"]
    test_data["protocol_type_udp"] = enc_protocol_df["protocol_type_udp"]
    test_data = test_data[headers]
    test_data["attack"] = test_data["attack"].apply(lambda x: 1 if x != "normal" else 0)
    test_data.drop(columns=["level"], axis=1, inplace=True)
    test_data = scaler.transform(test_data)
    test_data = pd.DataFrame(test_data, columns=header)
    test_data["class"] = test_data["class"].astype("int64")
    test_data = test_data[req_columns]
    return test_data
