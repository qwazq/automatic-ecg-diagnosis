clear

$fileWhere="D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_code"
$dataFileWhere="D:\codeBase\python\CrossCuttingIssues\automatic-ecg-diagnosis_data\test_data"

$hdf5Where=$dataFileWhere+"\"+"ecg_tracings.hdf5"
$modelWhere=$fileWhere+"\"+"final_model.hdf5"

python ".\predict.py" $hdf5Where $modelWhere