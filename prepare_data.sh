wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Digital_Music.json.gz
gzip -d reviews_Digital_Music_5.json.gz
gzip -d meta_Digital_Music.json.gz
python script/process_data.py meta_Digital_Music.json reviews_Digital_Music_5.json
python script/local_aggretor_time.py
python script/generate_voc.py
