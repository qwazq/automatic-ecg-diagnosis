clear
$path_to_hdf5="D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\test_data\ecg_tracings.hdf5"
$path_to_model="D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\model\model.hdf5"
$OUTPUT_FILE="./predictOutput"

python .\predict.py $path_to_hdf5 $path_to_model --output_file $OUTPUT_FILE

<#
cd D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_code
.\my_predict.ps1
 #>



