from bing_image_downloader import downloader
import os 
import numpy as np 
import re
import pandas as pd
query_string=["dosa","vada" ,"idly","all vegetarian food that should not include idly,dosa and,vada"]
output_dir='dataset-3'
for i in query_string:
    downloader.download(i, limit=50,  output_dir=output_dir, adult_filter_off=True, force_replace=False, timeout=120, verbose=True)
    path_to_directory= f"{output_dir}/{i}"
    csv_file_name=f"{i}_file_names.csv"
    file_name=[]

    # Iterate over the directory and append file paths to the list
    for filename in os.listdir(f"{output_dir}/{i}"):
        if re.match(r".+\.csv$",filename):
            pass
        else:
            file_name.append(f"{output_dir}/{i}/{filename}")

    
    # Create the output file in write mode

    data={"file_path":file_name,
          "value":[1 if i=="idly" else
                   2 if i=="vada" else
                   3 if i=="dosa" else
                   4 for _ in range(len(file_name))]}
    df=pd.DataFrame(data)
    if csv_file_name=="all vegetarian food that should not include idly,dosa and,vada_file_names.csv":
        csv_file_name="others.csv"
    output_file_path=f"{path_to_directory}/{csv_file_name}"
    df.to_csv(output_file_path,index=False)
    
    # with open(f"{path_to_directory}/{csv_file_name}","w") as outputfile:
    #     for file_path in file_name:
    #         number=1
    #         if re.match(r".+\.csv$",file_path):
    #             pass
    #         else:
    #             outputfile.write(f"{path_to_directory}/{file_path}\t{number} \n")


