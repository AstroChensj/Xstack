nh=$1
z=$2
lum=$3
gamma=$4
q=$5
sgamma=$6
ecut=$7
seed=$8
src_expo=$9
bkg_expo=${10}
spec_dir=${11}
pha_file=${12}
rmf_file=${13}
arf_file=${14}
out_pre=${15}
out_dir=${16}

fkpha_file=$out_pre".fits"
fkbkg_file=$out_pre"_bkg.fits"
fkarf_file=$arf_file
fkrmf_file=$rmf_file

new_fkpha_file=$out_pre"_src.fits"
new_fkbkg_file=$out_pre"_bkg.fits"
new_fkarf_file=$out_pre"_arf.fits"
new_fkrmf_file=$out_pre"_rmf.fits"
new_fknh_file=$new_fkpha_file".nh"
new_fkz_file=$new_fkpha_file".z"

cwd=$pwd

cd $spec_dir

model="tbabs*(clumin*zpowerlw+const*clumin*zcutoffpl)"
(
    echo model $model
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo
    echo 
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
    echo newpar 14 $sgamma
    echo newpar 15 $ecut
    echo newpar 16 $z
    echo newpar 17 1 -1 
    echo show
    echo 
    echo data $pha_file
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
    echo $fkpha_file
    echo $src_expo,1,$bkg_expo
    echo 
    echo exit
) | xspec

echo $nh > $new_fknh_file
echo $z > $new_fkz_file

mv $fkpha_file $out_dir/$new_fkpha_file
mv $fkbkg_file $out_dir/$new_fkbkg_file
cp $fkarf_file $out_dir/$new_fkarf_file
#cp $fkrmf_file $out_dir/$new_fkrmf_file
mv $new_fknh_file $out_dir
mv $new_fkz_file $out_dir

cd $cwd
