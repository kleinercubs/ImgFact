# Generate image-triplet pairs that fit in the requirements of different sub-task.
mkdir data
mkdir data/predict_o data/predict_s data/predict_p
mkdir data/predict_s/spo data/predict_s/messy
mkdir data/predict_p/spo data/predict_p/messy data/predict_p/s data/predict_p/o
mkdir data/predict_o/spo data/predict_o/messy data/predict_o/p
python s_spo_prepare_data.py --file predict_s/spo 
python s_messy_prepare_data.py --file predict_s/messy
python prepare_data.py --file predict_o/spo 
python messy_prepare_data.py --file predict_o/messy
python app_p_prepare_data.py --file predict_o/p
python p_spo_setup.py --file predict_p/spo 
python p_messy_setup.py --file predict_p/messy
python p_entity_setup.py --file predict_p/s
python p_entity_setup.py --file predict_p/o

# Generate image dictionary.
python image_naive_pkl.py
python image_vilt_pkl.py
python image_naive_pkl.py --type enhance
python image_vilt_pkl.py --type enhance

# Merge image dictionary and image-triplet pairs. 
python  vilt_prepare_data.py --dataset predict_s/spo 
python  vilt_prepare_data.py --dataset predict_s/messy 
python  vilt_prepare_data.py --dataset predict_p/spo 
python  vilt_prepare_data.py --dataset predict_p/messy 
python  vilt_prepare_data.py --dataset predict_p/s 
python  vilt_prepare_data.py --dataset predict_p/o 
python  vilt_prepare_data.py --dataset predict_o/spo 
python  vilt_prepare_data.py --dataset predict_o/messy 
python  vilt_prepare_data.py --dataset predict_o/p 

python  naive_prepare_data.py --dataset predict_s/spo 
python  naive_prepare_data.py --dataset predict_s/messy 
python  naive_prepare_data.py --dataset predict_p/spo 
python  naive_prepare_data.py --dataset predict_p/messy 
python  naive_prepare_data.py --dataset predict_p/s 
python  naive_prepare_data.py --dataset predict_p/o 
python  naive_prepare_data.py --dataset predict_o/spo 
python  naive_prepare_data.py --dataset predict_o/messy 
python  naive_prepare_data.py --dataset predict_o/p 
