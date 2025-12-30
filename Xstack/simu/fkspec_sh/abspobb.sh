#!/usr/bin/env bash
nh=$1
z=$2
lum=$3
gamma=$4
q=$5
seed=$6
src_expo=$7
bkg_expo=$8
spec_dir=$9
pi_file=${10}
rmf_file=${11}
arf_file=${12}
out_pre=${13}
out_dir=${14}

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

model="tbabs*(clumin*zpowerlw+const*clumin*zbbody)"
(
    echo model $model
    echo "/*"
    echo 
    echo newpar 1 $nh
    echo newpar 3 2 
    echo newpar 4 $z
    echo newpar 5 $lum
    echo newpar 6 $gamma
    echo newpar 7 $z
    echo newpar 8 1 -1
    echo newpar 9 $q
    echo newpar 11 2
    echo newpar 12 $z
    echo newpar 13=5
    echo newpar 14 0.1
    echo newpar 15 $z
    echo newpar 16 1 -1 
    echo show
    echo 
    echo data $pi_file
    echo 
    echo none   # because we may not have the rsp file
    echo none
    echo none
    echo 
    echo response $rmf_file
    echo 
    echo arf $arf_file
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

echo "$(echo "$nh * 10^22" | bc)" > $new_fknh_file
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
