# Generate image-triplet pairs that fit in the requirements of different sub-task.
python s_spo_prepare_data.py --file predict_s/spo 
python s_p_prepare_data.py --file predict_s/p
python s_o_prepare_data.py --file predict_s/o
python s_messy_prepare_data.py --file predict_s/messy
python prepare_data.py --file predict_o/spo 
python app_p_prepare_data.py --file predict_o/p
python app_s_prepare_data.py --file predict_o/s
python messy_prepare_data.py --file predict_o/messy
python p_spo_setup.py --file predict_p/spo 
python p_entity_setup.py --file predict_p/s
python p_entity_setup.py --file predict_p/o
python p_messy_setup.py --file predict_p/messy

# Generate image dictionary.
python image_naive_pkl.py
python image_vilt_pkl.py

# Merge image dictionary and image-triplet pairs. 
python  vilt_prepare_data.py --dataset predict_p/spo 
python  vilt_prepare_data.py --dataset predict_p/messy 
python  vilt_prepare_data.py --dataset predict_p/s 
python  vilt_prepare_data.py --dataset predict_p/o 
python  vilt_prepare_data.py --dataset predict_s/spo 
python  vilt_prepare_data.py --dataset predict_s/messy 
python  vilt_prepare_data.py --dataset predict_s/p 
python  vilt_prepare_data.py --dataset predict_s/o 
python  vilt_prepare_data.py --dataset predict_o/spo 
python  vilt_prepare_data.py --dataset predict_o/messy 
python  vilt_prepare_data.py --dataset predict_o/s 
python  vilt_prepare_data.py --dataset predict_o/p 
python  naive_prepare_data.py --dataset predict_p/spo 
python  naive_prepare_data.py --dataset predict_p/messy 
python  naive_prepare_data.py --dataset predict_p/s 
python  naive_prepare_data.py --dataset predict_p/o 
python  naive_prepare_data.py --dataset predict_s/spo 
python  naive_prepare_data.py --dataset predict_s/messy 
python  naive_prepare_data.py --dataset predict_s/p 
python  naive_prepare_data.py --dataset predict_s/o 
python  naive_prepare_data.py --dataset predict_o/spo 
python  naive_prepare_data.py --dataset predict_o/messy 
python  naive_prepare_data.py --dataset predict_o/s 
python  naive_prepare_data.py --dataset predict_o/p 
