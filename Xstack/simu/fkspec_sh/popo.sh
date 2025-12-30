z=$1
lum=$2
gamma=$3
q=$4
sgamma=$5
seed=$6
src_expo=$7
bkg_expo=$8
spec_dir=$9
pha_file=${10}
arf_file=${11}
out_pre=${12}
out_dir=${13}

fkpha_file=$out_pre".fak"
fkbkg_file=$out_pre"_bkg.fak"
fkarf_file=$out_pre".arf"

cwd=$pwd

cd $spec_dir

model="clumin*zpowerlw+const*clumin*zpowerlw"
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
    echo newpar 13 $sgamma
    echo newpar 14 $z
    echo newpar 15 1 -1 
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
