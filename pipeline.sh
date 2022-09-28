## NOTE: this script is mainly used for explanation. One needs to run step by step ##

################## Step 1.  Data Quality Control ###################
## 1. merge *bim *fam *bed files using plink into strata   
## merge bed bim fam files from different batches
plink --merge-list files.list --make-bed --out merged_all_step2_QCedSNPs

## 2. QC

plink --bfile merged_all_step2_QCedSNPs --geno 0.0 --maf 0.01 --hwe 1e-5 midp include-nonctrl  --make-bed --out merged_all_step2_QCedSNPs.qc1

plink --bfile merged_all_step2_QCedSNPs.qc1 --het --test-missing midp --pfilter 1e-4 --make-bed --out merged_all_step2_QCedSNPs.qc2 

## 3. to vcf
prefix=merged_all_step2_QCedSNPs.qc2
plink --bfile $prefix --recode vcf --out $prefix
bgzip $prefix.vcf
tabix -f $prefix.vcf.gz


################## Step 2.  Remove Batch Effects  ###################
python chi2_test.allele.py

################## Step 3.  Split data into training-test set  ###################

python split_dataset.py

################## Step 4.  GWAS for training data ###################

#remove testing samples
plink --bfile merged_all_step2_QCedSNPs.qc2 --remove-fam test.samples.fam   --make-bed --out merged_all_step2_QCedSNPs.qc2.rmtest

plink --bfile merged_all_step2_QCedSNPs.qc2.rmtest --assoc --threads 48 >gwas.log 2>&1 &


################## Step 5.  SNP Annotation using annovar (assign SNPs to genes) ###################

for i in {1..22};do perl  /software/annovar/convert2annovar.pl -format vcf4old  ../QC/chrs/chr$i.vcf >chr$i.avinput 2>/dev/null & done

for i in {1..22};do perl /software/annovar/annotate_variation.pl -out chr$i -build hg19 chr$i.avinput  /software/annovar/humandb & done

## reformat:
# if 'intergenic': get the nearest gene as the target gene.
# if annotated with multiple functions in different genes, assign each record one by one.
# if annotated with single function but multiple genes, assign each record one by one.
# output files: chr*.variant_function.uniqGene
for i in {1..22};do cat chr$i.variant_function|perl -ne 'my@a=split;my$gene; if ($a[0] eq "intergenic"){my@aaa=$a[1]=~/(\S+)\(dist=(\S+)\),(\S+)\(dist=(\S+)\)$/g;if($2 eq "NONE" ){$gene=$3;}elsif( $4 eq "NONE"){$gene=$1;}elsif ($2<=$4){$gene=$1;print "$a[2]\t$a[3]\t$gene\t$a[0]\n"} else{  $gene= $3;print "$a[2]\t$a[3]\t$gene\t$a[0]\n"} }else { if($a[0]=~/;/){my@funs=split/;/,$a[0];$a[1]=~s/\(\S*?\)//g; my@genes=split/;/,$a[1]; for my$i(0..$#funs){ my@subgenes=split/,/,$genes[$i];for my$g(@subgenes){my$gg=$g;$gg=~s/\(\S+\)//g; print "$a[2]\t$a[3]\t$gg\t$funs[$i]\n";}  }}else{ $a[1]=~s/\(\S*?\)//g;  my@subgenes=split/,/,$a[1];for my$g(@subgenes){my$gg=$g;$gg=~s/\(\S+//g; print "$a[2]\t$a[3]\t$gg\t$a[0]\n";} } } '|sort -u|sed 's/\//_/g'|sort -k2n -k3 -k4 >chr$i.variant_function.uniqGene ;echo chr$i;done


################## Step 6.  Perform Gene-PCA analysis ###################

##1. Get SNP dataframe for each gene

dir=/project/als/analysis/
for i in {1..22};do cd $dir/chr$i; python get_genes.py chr$i  & done

##2. Run Gene-PCA ###
cd $dir/chr$i; python gene_pca.py $chr $gene

##4. combined feature file
python combine_features.py

## run All-PCA
python all_pca.py


################## Step 7.  Train classifiers and make predictions ###################

#for Deep Learning , for both Gene-PCA & All-PCA
python capsule.GPU.py sigSNPs_pca.features.pkl 1.0 capsule_pca_1.0 1
python capsule.GPU.py allGenes_pca.pkl 1.0 capsule_allpca_1.0 1
python CNN.GPU.py allGenes_pca.pkl 1.0 cnn_allpca_1.0
python CNN.GPU.py sigSNPs_pca.features.pkl 1.0 cnn_pca_1.0
python MLP.GPU.py sigSNPs_pca.features.pkl 1.0 mlp_pca_1.0
python MLP.GPU.py allGenes_pca.pkl 1.0 mlp_allpca_1.0

#for basic ML, for both Gene-PCA & All-PCA
python basicML.py lr sigSNPs_pca.features.pkl 1.0 lr_pca_1.0
python basicML.py svm sigSNPs_pca.features.pkl 1.0 svm_pca_1.0
python basicML.py rf sigSNPs_pca.features.pkl 1.0 rf_pca_1.0
python basicML.py adab sigSNPs_pca.features.pkl 1.0 adab_pca_1.0
python basicML_allpca.py rf allGenes_pca.pkl 1.0 rf_allpca_1.0
python basicML_allpca.py adab allGenes_pca.pkl 1.0 adab_allpca_1.0
python basicML_allpca.py lr allGenes_pca.pkl 1.0 lr_allpca_1.0
python basicML_allpca.py svm allGenes_pca.pkl 1.0 svm_allpca_1.0

################## Step 8.  Determining core genes decisive for classification ###################
#see select_core_genes_for_classification.ipynb

################## Step 9.  Determining "non-additive" genes ###################
python select_nonadditive_genes.py


###### END #####

