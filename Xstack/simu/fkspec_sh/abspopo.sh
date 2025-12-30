nh=$1
z=$2
lum=$3
gamma=$4
q=$5
sgamma=$6
seed=$7
src_expo=$8
bkg_expo=$9
spec_dir=${10}
pha_file=${11}
arf_file=${12}
out_pre=${13}
out_dir=${14}

fkpha_file=$out_pre".fak"
fkbkg_file=$out_pre"_bkg.fak"
fkarf_file=$out_pre".arf"

cwd=$pwd

cd $spec_dir

model="tbabs*(clumin*zpowerlw+const*clumin*zpowerlw)"
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
    echo newpar 15 $z
    echo newpar 16 1 -1 
    echo show
    echo 
    echo data $pha_file
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

mv $fkpha_file $fkbkg_file $out_dir
cp $arf_file $out_dir/$fkarf_file

cd $cwd
