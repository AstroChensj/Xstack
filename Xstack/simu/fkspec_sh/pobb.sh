#!/usr/bin/env bash
z=$1
lum=$2
gamma=$3
q=$4
seed=$5
src_expo=$6
bkg_expo=$7
spec_dir=$8
pi_file=$9
rmf_file=${10}
arf_file=${11}
out_pre=${12}
out_dir=${13}

fkpi_file=$out_pre".pi"
fkbkgpi_file=$out_pre"_bkg.pi"
fkarf_file=$arf_file
fkrmf_file=$rmf_file

new_fkarf_file=$out_pre".arf"
#new_fkrmf_file=$out_pre".rmf"
new_fkrmf_file=${fkrmf_file##*/}
new_fknh_file=$fkpi_file".nh"
new_fkz_file=$fkpi_file".z"

cwd=$pwd

cd $spec_dir

model="clumin*zpowerlw+const*clumin*zbbody"
(
    echo model $model
    echo "/*"
    echo 
    echo newpar 2 2 
    echo newpar 3 $z
    echo newpar 4 $lum
    echo newpar 5 $gamma
    echo newpar 6 $z
    echo newpar 7 1 -1
    echo newpar 8 $q
    echo newpar 10 2
    echo newpar 11 $z
    echo newpar 12=4
    echo newpar 13 0.1
    echo newpar 14 $z
    echo newpar 15 1 -1 
    echo show
    echo 
    echo data $pi_file
    echo 
    echo xset seed $seed 
    echo 
    echo fakeit
    echo 
    echo 
    echo $fkpi_file
    echo $src_expo,1,$bkg_expo
    echo 
    echo exit
) | xspec

echo 0 > $new_fknh_file
echo $z > $new_fkz_file

fthedit $fkpi_file+1 keyword=BACKFILE operation=add value=$fkbkgpi_file
fthedit $fkpi_file+1 keyword=RESPFILE operation=add value=$new_fkrmf_file
fthedit $fkpi_file+1 keyword=ANCRFILE operation=add value=$new_fkarf_file
fthedit $fkpi_file+1 keyword=NHFILE operation=add value=$new_fknh_file
fthedit $fkpi_file+1 keyword=ZFILE operation=add value=$new_fkz_file
fthedit $fkbkgpi_file+1 keyword=RESPFILE operation=add value=$new_fkrmf_file
fthedit $fkbkgpi_file+1 keyword=ANCRFILE operation=add value=$new_fkarf_file

mv $fkpi_file $out_dir
mv $fkbkgpi_file $out_dir
cp $fkarf_file $out_dir/$new_fkarf_file
cp $fkrmf_file $out_dir/$new_fkrmf_file
mv $new_fknh_file $out_dir
mv $new_fkz_file $out_dir

cd $cwd
