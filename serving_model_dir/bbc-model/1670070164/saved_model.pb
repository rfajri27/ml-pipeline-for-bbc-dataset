ти
Є0╟0
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Р
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
б
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
.
Identity

input"T
output"T"	
Ttype
▄
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0■        "
value_indexint(0■        "+

vocab_sizeint         (0         "
	delimiterstring	"
offsetint И
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
TvaluestypeИ
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
TouttypeИ
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
TouttypeИ
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
и
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ

NoOp
М
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint         "	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Р
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
│
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ч
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
е
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint         
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
9
VarIsInitializedOp
resource
is_initialized
И"serve*2.8.22v2.8.1-10-g2ea19cbb5758╙╩
Е
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	РN*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	РN*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	└*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:└*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
└А*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:А*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	А*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
m

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name15764*
value_dtype0	
А
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_13818*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
З
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *1
f,R*
(__inference_restored_function_body_21285
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
Б
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
R
Variable/AssignAssignVariableOpVariableasset_path_initializer*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
З

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
X
Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer_1*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
У
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	РN*,
shared_nameAdam/embedding/embeddings/m
М
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	РN*
dtype0
З
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└*&
shared_nameAdam/dense_3/kernel/m
А
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	└*
dtype0

Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*$
shared_nameAdam/dense_3/bias/m
x
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes	
:└*
dtype0
И
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*&
shared_nameAdam/dense_4/kernel/m
Б
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
└А*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:А*
dtype0
З
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_5/kernel/m
А
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	А*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
У
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	РN*,
shared_nameAdam/embedding/embeddings/v
М
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	РN*
dtype0
З
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└*&
shared_nameAdam/dense_3/kernel/v
А
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	└*
dtype0

Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:└*$
shared_nameAdam/dense_3/bias/v
x
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes	
:└*
dtype0
И
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└А*&
shared_nameAdam/dense_4/kernel/v
Б
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
└А*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:А*
dtype0
З
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_5/kernel/v
А
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	А*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R
R
Const_6Const*
_output_shapes
: *
dtype0	*
valueB	 R
         
I
Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R
▐┬
Const_8Const*
_output_shapes	
:ОN*
dtype0*а┬
valueХ┬BС┬ОNBtheBtoBofBandBaBinBisBsBforBthatBitBonBsaidBheBbeBwithBwasBhasBwillBasBhaveBbutBatBbyBareBiBnotBhisBweBtheyBfromBthisBmrBanBtheirBbeenBwouldBalsoBwhoBhadBwereBupBwhichBmoreBitsBpeopleBthereByouBafterBusBoneBoutBifBaboutBnewBorBcanBoverBcouldByearBlastBwhenBnowBallBtBthanBtimeBtwoBwhatB
governmentBfirstBsomeBsoBintoBworldBukBgameBagainstBnoBmakeBbeingBotherBdoBgetBsheBnextBbeforeByearsBjustBbecauseBveryBsetBbestBmanyBshouldBmayBlikeBmostBelectionBmadeBfilmBtoldBwellBsuchBcompanyBtakeBenglandBthemBonlyBthreeBgoodBwinBthenBourBbackBwhileBoffBhowBanyBnetBlabourBwayBfirmsBsinceBpartyBplayBgamesBhomeBgoingBsayBfirmBworkBthoseBdownBaddedBbbcBthinkBhimBseeBmarketBcountryB2004BwantBmuchBpartB000BsystemBsaysBformerBeuropeanBblairBweekBsecondBendBmusicBministerBwonBusedBmyBunderBstillBcomeBplansBoBnumberBgroupBclubBbothBthroughBherBbetweenBpublicBnewsBneedBgoBbankBroddickBmBevenBbigBwithoutBtheseBlondonBcupBshowBmeBhoweverByourBuseBmatchBdidBmoneyBgeneralBduringBsixBserviceBplayersBfiveBbritishBalreadyB	accordingBusersBtopBmonthsBdoesBthirdBrightBgotBfinalBcampaignBbrownBbritainBtooBreallyBrealBteamBinternetBfutureBchiefBtvBputBmillionBdonB	broadbandBveBunitedBreportBopenBlostBfiguresBexpectedBdealBpricesBnadalBmoveBmobileBknowBdirectorBwhereBspendingBseenBplaceBmonthBcurrentBaroundB	secretaryBindustryBfourBeconomyBservicesBissueBfoundBforeignBdayBcalledBagainB10BwhetherBtoryBownBhitBgrowthBfaceBeconomicBsoftwareBsayingBplayedBlawBirelandB	includingB	companiesBstateBpointBlikelyBlegalBinternationalBhowardBgreatBfewBwentBvideoBsecurityBrecordBmsBlotB	importantBearlyBdataBcameB	statementBrobinsonBplayerBhouseBgivenBearlierBcourtB2003BuntilBtookBprimeBdueBbiggestBtrialBtitleB
technologyBsideBreturnB	presidentBphoneBmightB	liverpoolBeveryBcomesBbudgetB2BwomenB	spokesmanBofferBlessBclearBanotherBamBusingBstartBresearchBincreaseBfarBeuropeBdespiteBbusinessBtaxBpayBnationsBlordBlifeBinformationBindiaBheldBhardBdrBdoneBdollarBcutBweeksBtakingBroleBpartiesB	microsoftBinterestBholdBhereBfranceBfindBchangeB
chancellorBcannotBbreakBbetterBbehindBwalesBtuesdayBthoughtBsureBsameBrulesBoilBofficeBneededBmarkBmarchBlongBiraqBeverB	differentBcoachBchoiceBcallsB	announcedBaheadBsummerBsquadBsiteBreBproblemsBlookBleaderBlatestBjanuaryBhalfBfurtherB	followingB	difficultBdBchinaBchanceBbecomeBbandBamongByetBwinningBvictoryBproblemBplayingBnationalBmondayBleadBitalyBinjuryBhopeBfrenchBformBdecemberBcontrolBclaimsBbeatBalthoughBactBtimesBsupportBseasonBsaleBprogramsBlineBhighBdaysBdavidBchairmanBagreedB2005BwithinBwarnedBwantsBversionBrunBpointsBpastBmenBmeetBmainBleftBkeyBirishB	executiveBenoughBdecisionBcityBallowBactionBacrossBaccountsBvotersBtennisBsaturdayBphonesBpersonalBmichaelBmembersBkeepBhistoryBhelpB	financialBeventBcontentBcompetitionBbillBawardsBamericanBalwaysB2002B20BworkingBwinnerB	wednesdayBwarBvoteBviewBsundayBsmallBsalesBrugbyBrecentlyBrecentBquiteBperformanceBmeanBmanBlowBlittleBledBinsteadBfridayBensureBdigitalBclarkeBbookBbelieveBalmostBadmittedBableBtryB
themselvesBtakenBsirBserveBroseB
parliamentBnovemberBminutesBmediaBllBlionsBlevelBleastBlaterB
kilroysilkBjohnBhavingBgiveBgettingBfullB	educationBeachB	customersB	creditorsBcouncilB	consumersBcentreBanalystsBtourBtonyBsystemsBsportB	somethingBruleBriseBpriceBpoliticiansB	politicalBpolicyBpcBothersBneverBmpsBmomentBmakingBmadridBlocalBimmigrationBhealthBheadBforwardBforceBfebruaryBcashBappleBtryingBtogetherB
televisionBstrongBsitesBseriesB	programmeBonlineBmeansB
manchesterBmajorBissuesBinvolvedBfreeB	currentlyBcostBbtBbidBasylumBannualBagoBaccessB2001ByahooBwholeBwebsiteBulsterBukipBspecialB	situationBshowedBschoolBroundBreportsBratesBrateBradioBpressureBneedsB	ministersBmikeBfactBeurosBeuBeightBcomputerBchampionB	argentinaB5BwilliamsBwantedBtodayBthoughBsharesBrevealedBresultBraceBquarterBpowerBpensionsBparentsB	newcastleBnameBlordsBlookingBlibBletBlaunchBkellyBgrowingBfrontBdidnBcontinueBconsumerB
conferenceBbushB17B12BwebBtoriesBthingsBsuccessB	proposalsBpossibleBplanBonceBmustBleaveBjobBjapanB
investmentBindependentBincludeBfourthBfansBfallBeuroBenergyBdoingBcrimeBcourseBcostsBcomparedBchelseaBarsenalBwirelessBwinsBvisitorsBtotalBstockBsoldBpoorBorderBnothingBmirzaBmakesBloveBlightBleadingBlawsBjuneBindianBideaBhugeBfellBfeelB
experienceBevidenceBedwardsBdeniedBdebtBcriminalBbuildingBbossBbanksBawardBaskedBanythingBaidB2000B11B100ByoungBwhyBtrustBtermsBstatesBstaffBsentBschoolsBreportedBprovideBpreviousBpressBpopularBpoliceBofferedBnetworksBmorganBmeetingBmeasuresBliberalBlateBjumpBincludedBhappyBgordonBgermanyBfilmsBfestivalBdeathBcutsBcrisisBchildrenBcaseBboostBboardBbitBbelievesBaverageBareaB4BwrittenBwinnersBviaBthingBstreetBstarBsouthBshotBseveralBruledBrockBroadBremainBreleasedBratherBprivateBpovertyBpanelBmovieBmindBlivesBhuntingBgoldenBglazerBgermanBgaveBfraudBfinallyBfightBfamilyB
everythingBdemandBcreateB
commissionBbattleBbanBbadBawayBassociationB3B1997B18BwaitingB	transportBtrainingBtradingBtoughBtimBthursdayBtestBtalksBscoredBrobertBrightsBreleaseBreceivedBreachedBraiseB
programmesBoctoberBmunsterBmessageBmanagerBliveB	leicesterBjulyBjohnsonBhostBhenryBgroupsBgoalBfinanceBextraBeitherBdefenceBcomingBcodeBclaimBchangesBcampbellBbeganBbasedBballBattackBareasBannounceBaddB25B16BvehiclesBvalueBturnB	suggestedBstageBspainBsoonBsocietyBsevenB	septemberBsearchBrunningBresponseBrespondBreachBprogressBprocessBpremiershipBpollBperiodBparticularlyBpaidB	officialsBnetworkBnamedBmissingBmemberB	meanwhileB	marketingBmachineBleagueBkeenBhumanBhopesBhimselfBhenmanBgrandBgmtBfundBfollowedBfederalBdevelopmentBdebutBdanceBcrowdBcontractBconsideringB	communityBcaptainBbathB	availableBanswerBannouncementBafricaB	advantageBactressBactorBaccountB30B1BzealandBwaysBvirusesBtrafficBteamsB
successfulBstrikerBstandBspamBspaceBsimplyBshowsBshareBseniorBscotlandBrowBprofitBprizeB	potentialBpoliticsBpaulBoftenBofficialBmurrayBkeptBitselfBimproveBianBhoursBhalflifeBgoneBglobalBfreshBforcedBfootballBfailedBeveryoneBemailBdrugBdownloadBdemsBdavisB	criminalsBcommonsB	committeeBcloseBcharlesB	challengeBbroughtBblockingBaugustBaprilBapproachBappealBandrewBageBadviceBachievedB40B15B14BworksBwidelyBvanBvBtradeB
throughoutBtheatreB	technicalBsteveBsonyBsignedBshownBseriousBsecBriskBrichardBremainsBquitBqueenBqualityBproposedBplanningBpiracyBpenaltyBopportunityBoldBnumbersBnineBnetcraftB	maliciousBlistBleinsterBlaBknownBjpBjonesBjBhappenBgroundB
generationBgasBfocusBfinishedBfearBfairB
executivesBeditorBdriscollBdoesnB
departmentBdefeatBdebateB	countriesBcompeteBcommentsBcasesBcarBbrianBboughtBbondBbettingBbenitezBbasicBavoidBandyBanalystBallowedBalanB75B50B24B2006B13BwritingBworthBwideBvotedBumagaBturnedBtrueBtowardsBteachersBtaxesBtakesBstayBstartedBstandardBspeedBspeakingBsingleBsingerBsignBsellBseedBsaveBrussiaBrejectedBputtingBpromisedBprisonBpaceBnintendoBnearBmobilesBmilitaryBmatterBmajorityBmagazineBmachinesBlycosBlevelsBlargestBlargeBlackBjoseBjobsBinsistedBinquiryB	increasedBillegalBhomesBhigherBharryBgerrardBgeorgeBfindingBfigureBfieldBfacesBexpectBeffectBeasyBdriveBdesignedBdateBcopiesBconservativeB	confirmedB	confidentBconcernB	championsBceremonyBcareBcallBblackBbiggerBbelowBbelievedBbasisB	barcelonaBauthoritiesBargonautB	argentineBacceptedB62ByorkB	wimbledonBtaylorBtalkBstepBsportsBrogerBreasonBreadyBproperB
productionB	producersBpowerfulBpotterB
pensionersBpensionBoutsideBopinionBoperaB
multimediaB
monitoringBmilburnBmetBmessagesBlineupBjamesBipodBholdsBheartBfeaturesBfastBfacingBfacedBexpertsBexampleB
especiallyBdoubtBdetailsBdecideBcriticalB
consideredBconsiderB	computersBcommunicationsBcommentBcollinsBchampionshipsB	certainlyBcentralBbuyingBbuyBbecameB
australianBapprovalBalbumB	agreementBaffectedBadministrationB7B64B31BworriedBworkedBwomanBwindowsBwhoseBwaterBwatchB
vulnerableBveritasBtriesBtindallBthreatB	telephoneBtargetBsuspectsBsullivanBstudyBstressedBstadiumBspentBslaveryBshadowBsawBsavingsBsanBrisingBrestBreplaceBprovingBpropertyBprojectBproductsBplannedB
particularBparmalatB	organisedBolympicBmpBmoyaBmissBminiB	maternityBmapsBmacBlosingBlongerBlistingsBlibraryBlendingBlegislationBlaunchedBkingBjusticeBjudgeBjoinedBimmediatelyBhopingBgreaterBgivesBgazpromBfineBfeltBfallingB	explainedBendedBdutyBdrugsBdroppedBdoubleB	directiveBdevicesBdesignBdemocratBdecidedBdeBdatingBcoupleB
confidenceB	concernedB	commodoreBcomedyBcoleBchargesB	characterBchangingB
challengesBcardsBcanadaBborussiaB	borrowingBbodyBblunkettB
birminghamBbaileyB	australiaBattacksBanyoneBangryBagentB63B60BwrongBworkersBwordBwhiteBweakBwatchdogBvoiceBvalveBunlikelyBtouchBterrorBtermBstrengthBstopBstickBspaniardBsomeoneBsidesBsharksBshareholdersBsendBsecondsBsafeBrichBreviewBrevenueBresponsibilityB	remainingBreformBreadingBquicklyBpushedBpupilsBprovedBprobablyB	predictedBpositionBpoliciesBpersonBpeaceBparkBoverallB
oppositionBopeningBolderBofferingB	newspaperBmoviesBmissedB	materialsBmassiveBmartinB
managementBlooksBlongtermBlistsBleavingBjuryB
interestedB	instituteBhighlyB	happeningB
gloucesterBgiantBfunBfollowBfiscalBfightingBfasterBentertainmentBearningsBdrawnBdramaBdocumentaryBdocumentB	democratsBdeadBcontroversialB	contractsBconservativesB
connectionBconcertBconcernsBcommonBclosedBcitizensBchargeBcelticBcategoryBcareerB
businessesBbrokeBboxBaviationBautumnBattorneyBappearB	alongsideBalongBairBagreeBagendaBaffairsBabroadB8ByukosBwoodwardBwestBweekendBvirusBvinciBvillaBveteranBunveiledBunlessBtwiceBtrackBtargetsBtakeoverBsubjectBstudiosBstrawBstraightBstevenB
statisticsBstartingBspywareBslamBsimilarBsiliconBsigningB	sensitiveBsenseBseemsBseekBscreensaverBscreenBrupeesBroyalBronaldoBrivalsBrisesBrfuBregularBreadBrangeB
protectionBprofitsBproductBpowersBpopBpersieBpcsBpassBpartlyBparisBownersBoverseasBoutputB	opponentsB	operatingBnhsBmyselfBmusicalBmortgageBmooreBmonsantoBmillionsBmilanBmassBmarutiBmarketsBlowerBlinkedBkaplanB
introducedBindoorBimprovedB
impressiveBholmesBhitsBhelpedBheavyBhappenedBhandheldBhandBgoogleB	goldsmithBgoldBgoesBgivingBgetsBgeBfriendsBfounderBfedererBfedBfearsB	favouriteBfailureB	estimatedB	electionsBdsBdortmundB	documentsBdirectBdeviceBdeliverBdealsBdawsonBcrossBcriticsB
criticisedB	continuedBcontestBconsoleBconnorsBcompleteB	competingB	committedB
commercialBclearlyBclashB	christmasBcheapBchancesBbrowserBbringB
bogdanovicBbillionBbellBaxaBattemptBathensBassemblyB	apologiseBamericaBallowsBaimBaguasBaddressBaccusedB
absolutelyBabilityB60mBworseBwebsitesBwaspsBwasnBvioxxB	valentineBunionBtroubleBtriedBtrendB
tournamentB	tomlinsonBtechnologiesBtechBtaskB	suspendedBsupplyBstyleBstrategyBstationsBstarsB	standardsBstBspokenBspeechBspeculationBspammersBsoundBsmallerBskypeBsixthBsimonBshowingBshortB	sheffieldBsettingB	seriouslyBseesBseekingBseeingBsecureBsectorBseatBrumoursBrooneyBrivalB	resourcesBrepublicBreduceBranB	questionsBquestionB	publishedBproducedBpresentBpreparedBpositiveB
populationBplaysBpictureBpickedBpeterBpernodBperBpaymentsBpayingBpaulaBpatientsBpackageBownsBoriginalB	operatorsBoperateBnorthBnationBnastyBmurphyBmunichBmedalBmaryBmailBlossesBlookedBlockBloanBlanguageBitalianBissuedB	interviewBinterestingBintelligenceBinitialBincludesB
impressionBimpactBhoullierBhandsBhandedBguysBgreenBfoxBfinesBfilesBfavourB	extremelyBexpectationsBexistingBexchangeBeventsBeveningB	efficientB	edinburghBedgeBeastBearnBdubaiBdrawBdomesticBdivisionB
differenceB
developersBdetailedBdeclaredB	decisionsBdecadeBcreditBcreatedBcoreBcookB	continuesBconsolesB	conductedB
complianceBcollapseBclaimedBchosenBcharityBchannelBchangedBcertainBcebitB	carpenterBbuildBbrandBblameB	blackpoolBbackedBasideBartistsB
argentinasBantispamBaddingBabsaB6B2007B	18yearoldB100mByenBwritersB	workforceBwordsBwilsonBwhateverBwengerBwallBvolleyBvisitBviottiBtruthBtrustedBtoolBtiebreakB	thousandsBthinksB	terrorismBtaitBtacticsBtableB
supportingBsuggestBsubstantialBstandingBstakeBspreadBsportingBspokeswomanBsortBsongBsolutionBsocalledBsmithBslavesBsevensBsettleBsellersB	secondaryBrussianBrodgersBrobinBresponsibleBrescueBrequiredB	regularlyBredBreceiveBreasonsB	qualifiedBputsBpushBproposalBprogramBproduceB
previouslyBpatentBpartsBparryBownerBowenBoutstandingBosbourneBordinaryBontoBonesBoffersB	obviouslyBobviousBnorthernBnominationsBnightBmovingBmovedBmorningB	morientesBmomentsBministryBmedicalBmatthewBmattBmaterialBlotsBloseBlinuxBlinkBlimitBlikesBletwinBletsBlearyB
leadershipBlawsuitBlaserBlargerBlaneBkronorBknowsBkirchnerBkindBjunkBjuninhoBjonathanBjoiningBjasonBjailBjackB	investorsBinvestBintendedB
individualBimfBibmBhuttBhousingBhourBholdingBholdersBhewittBhavenBhandsetsBguiltyBgrowBgibbonsBgamingBgBforecastBexplorationB	expansionBexcitingBenvironmentBenormousBenglishBendsB	employeesBeffortB	effectiveBeasierBeaBdvdBdropB
downloadedBdowningB	dismissedB
disciplineBdirectlyBdiedB	developerB	developedB	detentionB	detaineesBdetainedB	desperateBdenialBdeclinedBdeclineBdebtsB	davenportBcycleBcuttingBcreativeBcreatingBcouldnBcontributionB
continuingBconnectionsB
commitmentB
colleaguesBcloserBclassesBcauseBcatchBcarryBcardinalBcardBcapacityB
candidatesBcanalBcameraBcableBcabinetBbuiltB	broadcastBbreaksBbornBbordersBbenefitsBbellamyB	behaviourB	beginningBbayernBbarclaysB
bankruptcyBbaghdadB
archbishopBapplicationBappearedBanfieldBamountBalternativeBallowingB	addressesBachieveB9B80B70B61B21B1998BzoneBwroteB	worldwideBwillisBweaponsBwaveBwBvotingBversionsBveraBurgedB
universityBunitBunemploymentB
understandBtsunamiBtrophyBtoppedBtomBtitlesBtippedBthemeBthanouBtellBtelekomBtapesBswitchB	sustainedB	surprisedB
supportersBsupplierBsuperbBsuicideBsuggestsB	sufferingBsufferedB
subsidiaryBstruckBstronglyBsteamBstagesBspecificBspanishBsounessBsortsBslippedBskillsBsinglesB	singaporeBshapeBscrutinyBschemeBsceneBsavesB	satelliteB	santanderBrushBrunupBruddockB	rossignolB
retirementBrequestsB
representsBrelianceBrelationshipBregimeBrefusedBrefereeBreducingBreducedBrecoveryBrecordedBraisedBquotasB
questionedBpspBprotectBpropBprofessionalBprinceBpostedBpostBpossibilityBportalBpoppinsBpledgeBpleasedBplatformBplasticBplacesBplacedBpiratesBpesosBorganisationB
optimisticBolympicsBoatenBnovelBnorBnokiaBniesrBniceBnervousBnamesBmonstersBmonitorBmodelsBmodelBmistakeBminorityB
midfielderBmerckBmeasureBmeantBmanagingBmakerBlossBloansBljubicicBlinesBlimitedBleadersBlayBkevinBkenBirbBinvolvementB	interestsB
intentionsBintelBinfrastructureB	inflationB	improvingBimageBhurtBhugelyBhotelBholeBhitechBheadedBhardlyBgwB	guitaristBgrantedBgardenerBgangsBgalleryBg8BfullyBfuelBfrostBfootBfoodBfitBfinedBfinancesBfccBfamousBfailingBfaBextendB	expensiveB	excellentBethnicBentitledB
electronicBelderlyBeffectivelyBeadsBdutchBdunneBdrivenBdrakeBdownloadingBdoubledBdohertyBdisappointedBdirectedBdevelopBdeputyBdemB	deliveredBdefendedB
cunninghamBcrucialB	criticismB	corporateB	copyrightBcopyBcontrolsB
consistentBconnellBconnectBconfidentialBconceptBcompetitiveBcliveBcivilBchrisBchineseBchesterBcausedBcarlosBcapitalBbringingBbribeBbotinBboostedBbooksBbeyondBbenefitBbenBbeatenBbacksBbaaBauthorBattemptsB	attackingBaskingBaskBashdownBartsBarrestBarguedBarcherB	appointedBalexBairportBafricanBactuallyBactivityBaboveB500B3gB32B29B28B27BzambiaByuganskneftegasByesBwriterBwrightBwinterBwingerBwillingBwilliamBwildBwhilstBwestminsterBwelcomeBwasteBwarningBwakeBvisaBvehicleBusefulBupgradeB	universalBunacceptableBturnsBtruantsBtripBtpsBthreatsBtestedBtellingB	telephonyBtelecomsBtelecommunicationsBteenageBtargetedBtalentedBtalentBtackleBsydneyB	suspectedBsurveyBsunBsummitBsummaryBsuggestionsBsuezBsuedBsuddenlyBstuffBstudiesBstuartB
strugglingB	structureBstoryBstolenBsteppedBstealB	stabilityBspotBspendBsonBslightlyBsimpsonBsilvaBsignificantB
seychellesBsetsBservesBserenaBsequelBsellingB
securitiesBsecuredBscrumBscottBscoreBscienceBscarletsBsafinBsaddamBrunwayBrubinBrubberBreturnedBretiredBretailBresultsBresignedBrepresentativesBreplayBremoveBremarksBreformsBreflectB	recoveredBrecalledBreadersBrawBrasheedBrafaelB	radcliffeB
qualifyingBqcBpurchaseB	providingBproviderBproveBproudB
propertiesBpromiseBproBprettyBpresentsBpremierBppiB
portsmouthB
popularityBpledgesBpiresBphysicalBphilBpharmaBpersuadeBperhapsBperformancesB
peertopeerBpaxBpassedBparliamentaryBpaperBpanicciaBownedB
originallyBorganisationsBordersBoptinrealbigBobligesBnorthamptonBnooneBnobodyBneitherB
nationwideB	nationalsBmourinhoBmotorolaBmorphemeB
millenniumBmigrantsBmiddleB	messagingB	mauritiusBmathewBmartinezBmanicBmanagesBmalwareBlistingBlinksBliftBliesBletterBleighBleeBlearnBlawyerB	landscapeBkongBknockedBjuventusBjumpedBjoyBjonnyBjointBjoinBjapaneseBjacksonBitunesBinvestmentsB	integrityBinsideB
innovationBinitiativesB	initiallyB	influenceB
industriesBincreasinglyB
increasingB	increasesBimposeB
importanceBimplementationB
immigrantsBimagineBieBideasBhurdlesBhuntBhospitalB	hollywoodBhillBhighestBhideB
hemisphereBheathrowBhearB	guaranteeBgraphicsBgaryBgaraBgadgetsBgadgetBfundsBfullerBfreddieB	forecastsBflynnBflyingBfloorBfirefoxBfifthBfewerBfeniceBfellowBfeelsBfeelingBfaithBfactualBexplorerBexploitBexpectsB	everybodyB	essentialBericssonB	equipmentBenvironmentalBensuringBelseBellisBelectronicsB
economistsB	economistBecBeasilyBdublinBdrogbaBdoorsB	donaldsonBdomeBdollarsB
disruptiveBdisplayBdiscussionsBdisasterBdisappointmentBdisappointingBdisappearanceBdesireB	delightedBdelaysBdefendBdeepBdarkBdailyBdaBcustomerBcustodyBcurtisBcurrencyBcrownB	creaturesBcopyingBcontroversyBcontrastB
componentsB
completelyBclubsBchooseBchapterBchampionshipBchambersBcarrollBcarriedB	carefullyBcamusBcampaigningB	cambridgeBbroadcastersBbritonsBbreakthroughBbrandsBboysBblowBbloggerBbidsBbennettBbegunBbattlesBbaseBbarrierBbarkleyBbaezBbabyBbBawareBautomaticallyBaustriaBauditBaudienceBattractBathleteBasianBarrivedBarcyBapprovedBapartB
antiterrorBankleBanimatedBallianceBallegedBagencyBafiBadvancedBadmitBadditionBactsBaceB	abandonedB76B45B2008B200B1999B1982B1980sB┬г6mByourselfBwriteBwrightphillipsBworldcomB	wilkinsonBwelshBweinerBweatherBvodafoneBvisualBvisionBviolenceBviewersBvastBvariousBusuallyBusualBusesBuserBunveilBunionsB	uncertainBuncappedBuefaBtypeB
twickenhamBturningB
turnaroundBtroubledB
tremendousB	treatmentBtreasuryBtravelBtraditionalBtotallyBtoolsBtommyBtitanBtiesBtieBthievesBthierryBtextBtestsBtenBtellsBteenagerBtanziB
successiveB
substituteBsubstantiallyBsubscribersBstudioBstringerBstraightforwardBstarringBspringBsporeBspeedsBsourceB	sothertonBslowBskipperBsitBsimpleB	sidelinesBshrekBshopsBshootingBshockBsharpBsevereBservedBsentenceB	semifinalB	selectionBseemBseekersBsectorsBsectionB	scheduledBscandalBsativexBsabbathBruthBrunsB	runnersupBrulingBrowlingBrisksBrisenBrioB	ringtonesBrevenuesBreturnsBresolveBrequirementsBrequestB	reportingBreliefBrejectB
registeredBregionalBregionB
reelectionBredsBrecoverBrecordsBrecommendedBraisingBrafaBrBquotaBqualificationsBpushingBpuntersBpublishBprovidedBprovesBprosecutorsBprojectsBproducerB	processorBprizesB	presentedB	preparingB	preachersBpotentiallyB	portfolioBportableBpompeyBpolicingBplentyBphishingBphillipsBperformB	penaltiesBpeakBpbsBpatternsBpatchesBpassesB	passengerBpartnerBparticipationBpairBpainBpackBpB	ownershipBoptionBopposedBopportunitiesBopensBopenedBongcBolivierBoliverBobservedBnoticeBnormalBnoonB	nominatedBnecessarilyBnaturalBnaoBnannyBmuseumBmouseBmonetaryBmobilityBmmsBmetalBmergerBmentalBmelBmeasuredB	mcmanamanBmaximumBmattersBmatchesBmarathonB	manifestoBmakersBmaintainB	mackenzieB	macdonaldBltaBlivingstoneBlistenBliningBlimitsBlightingBliamBlatinBlarkinB	lansdowneBknockoutBkeeperBkayBjudgingBjudgesBjournalistsB
journalistBjoinsBjoblessBjamieBitvBiraqiBipv6BinvestigateBintroducingBinternalBinstanceB
inevitableBindexBindependenceB
incrediblyBincrediblesBincomesBimprovementB	impressedBimposedB	illegallyBietfBidolBiaafBhundredsBhousesBhotspotsBhostsBhongBhonestBhodgesBhockneyBhiddenBhickieB	headlinesBhandleBguardianBgrossBgrewBgregB	greenwoodBgreatestBgrahamBgoalsB	generallyBgbBgavinBgatheredBgartnerBfriendBfreezeBforgeardBflowBflankerBfixesBfinishBfinanciallyBfeatureBfearsomeBfdaB
favouritesBfaultBfashionBfanBfamiliesBeyesBeyeBextrasBextentB	expressedBexportsBexactlyBeverydayBevertonB
eventuallyBestablishedBenterBenduredB	encounterB	employersB	elsewhereBelementBeditingB	economicsBebbersBeasterbyBdrummerBdrmBdrivingBdraftBdozensBdotcomBdomecqBdistributionBdiseaseB	discoveryB	disappearBdiplomaBdipBdevaluationB	describedBdemonsBdefenderBdecidingBdangerBdamagesBcultureBcrudeB
crosscourtBcowellB
conventionB
contributeBconstituencyBconstantBconsolidationBconsecutiveB	completedBcommitBcommissionerBcombinationB
collectionB
collateralB
cochairmanBcmlBcloselyBcleanB	classroomBclassBchiracBchipBchaseBchallengingBcentresBcdBcbsBcaughtBcannabisBcampBcallingBbusBburrenBbrowsersBbrowseB	breakfastBboyB	bortolamiBbookedBbonusBboeingBbluesBblockBblacksBbillsBbillionsBbernardBbeninBbeginsBbeginBbecomingBbecomesBbeatingBbattledBbarosBbannedBbackingBawesomeBawardedBavoidedB
attractiveB
attractionBattitudeBassetsBaspectsBarticleBarguesB
appearanceBapparentB
apologisedBapacsBanybodyB
announcingBanilBangolaBangelsBamroBamidB	americansBalliedBalcoholBalbumsBakamaiBaircraftBairbusBaimingBaimedBagricultureBagreeingB
afterwardsB	aerospaceBadvanceBadoptedBadmitsBadministratorsB
additionalBacquisitionsBacknowledgedBacceptBacBabuseBabnBaaasB81B55B52B46B44B36B35B300B26B250B1994B1970B┬г50BzombieBzeroByuganskByoungerB
yearonyearByappBwouldnB	workplaceBwishBwindowBwerderBwelfareBwatchingBwaryBwantingBwallsBwaitBvowedBvisitorBvisitedBvirtualB
violationsBvieiraBvictorBvictimsBvickeryBvenueBventureButilityB	utilitiesButdBusaBuponBuploadedBupbeatBunusualBunlikeBunitsBunfortunatelyBunderstandingBundergroundBunableBunBtutuBtripleBtrioBtribunalBtrendsBtransferBtragedyBtoyotaBtoureB
tighteningBtightBtiedBticketsBthrillBthomasBthirdlyBthinkingBthemesBtheftBthanksBthankBtgwuBtestingB	terroristBteammateBteacherBtariffBtalkingBtacklingBswitzerlandBswissB	suspicionBsurgeryBsurgedBsurfaceBsuperBsundanceB
suggestionB
suggestingB
sufficientBsufferB
successionB	successesBsucceedB
subsequentBsubscriptionBsubsBstudentsB	struggledBstrongerBstrokeBstrikingBstreetsBstowellBstoreBstorageBstoodB	stimulateBstiffBstepsB	steinmetzBstatusBstationBstartupBstanstedBsprinterBsplitB
speculatedBspecificationsB
specialistBspeakBsouthernBsouthamptonBsorryBsomebodyB	solicitorBsmsBslugBslashedBsizeBsightB	shorttermB	shortlistBshoppingBshaunBshaneBsevilleBseventhBseparateBsendingBsenatorBsenateB
semifinalsBseemedB	scrumhalfB	screeningBscopeBscholesBschemesBscareBscandalsBscanBsavvyBsavedBsatisfactionBsarbanesoxleyBsaniaBsamoaBsalamBsafetyBsadlyBrubbishBroutineBroomsBroofBromeBrollingBroboticBrichterB	returningBrethinkBrestructuringBrespectivelyB	resigningBresignationBrequireBrepresentedB	representBreplacementsBreplacementBreplacedBrennerBrenewB
remarkableBremainedBrelyB
regulatoryB
regulatorsB	regulatorBrefersBrankedBquinlanBquincyB
quiksilverBquickBqualifyB
qualifiersBpullB	publisherB	publiciseBpublicationB	providersBprotocolBprosecutionsBproducesB	presenterBpreparationBpreelectionBpredictsB	prebudgetBpraisedB	practicesB
postponingBpossiblyBportoBportB
politicianBpointedBplusBplumbBpleadedBpitchBpickingBpickBphoenixB	petroleumBpetrochemicalsBpeteBpersonalityB	performedB	perfectlyBperfectBpeerBpayoutsB
paulsmeierBpatchBpassionBpassingBpartnershipBparticipateBpalaceBpageBovercomeBoutlinedBoscarB
organisersBorganiseB
operationsB	operationBopenerB
olympiakosBoffsetB
officiallyBofficesBofficerB	numericalBnoteBnoneBnomineesB
newspapersBnevilleBneillBneilB	negotiateBneatBnearlyBnatalieBmyskinaBmukeshBmovesBmovementBmothersBmotherBmoodBministerialBminimumBmidfieldBmercuryBmepBmeetingsB	mccormackBmccallBmarkedBmarieBmariaBmarginBmanufacturingBmanagedBmagicBlupoliBlooseBlivingBliteraryBlibertyB
leverkusenBlettingBlengthBlendersBlawyersBlBknowingBkneeBknapmanBkluftBkimBkilledBkickingBkickBkeyboardBkerrBkerekouBkennedyBkeepingBkeeganBkeaveneyBkeaneBjustifyBjupiterBjohnnyBjimmyBjezBjewishBjeremyBjanetBivanBitaliansBisraelBislandBirregularitiesBiraqisBipBinviteBinvestigationsBinvestigationBinvestedB
inventionsBinternazionaleBintenseB	insistingBinjuriesB
inhibitorsBinfectB
industrialBindefinitelyBindeedB	indecencyB
incredibleBincomeB	immigrantBimagesBhuntsB
housewivesBhotspotBhorganBhopedBholyBholesBhistoricBhingisBhighlightedB	highlightBhelpsBhelpingBhellBheinekenBheavilyBhearingBheardBhazellBhasnBhardwareBharderB
hantuchovaBhandsetB	halfbloodBguyBguitarBgrowsBgrownBgrewcockBgreekBgoodsBgoghBgladBgirlBgibsonBgiantsBgenerousBgenerateBgcsesBgatwickBgamersBgallasBgainBfundamentalB	functionsBfullbackBfulhamB	franciscoBfoundedB
foundationBforwardsB	fortunateBformsBforehandBforcesBfollowsBfocusedBflyhalfBflawsBfixedBfixBfiveyearBfirmlyBfiringBfiredBfireBfindsB
filmmakingB
filmmakersBfillBfilesharingBfifaB	ferdinandBfeetBfatherB	fantasticBfallsBfairlyBfailuresBfacilityB	extensiveBexplainBexpertBexpenditureBexcitedB	exchangesB	evolutionBeventualB	estimatesBerrorsBeraBequippedBequalBepisodesBentryBentrepreneurBensuresBengageBencouragingB
encouragedBemergedBemergeBeltonBelectedBeffectsBedwardBedgwareBeasternBeaseBduffBdreamBdraytonBdramaticBdownturnBdoublesB	dominatedBdoctorB
disruptionBdisputeBdisillusionedBdiscussBdisabledBdirtyB	directionB
developingBdeutscheBdeterminationB
detentionsBdepositBdepartmentsBdennisBdemosBdemonstrationsB
definitelyBdeficitBdefencesBdefaultsBdeciderBdeathsBdeadlyBdamienBdamagedB
criticismsBcreatesBcrashBcox2BcoveringBcostlyBcostarBcorporationBcorpBcornerBcopeBcoolB	convictedBconvertBcontainBconstructionBconstituenciesB
consortiumBcongressBconfirmBconductB
conditionsB	concludedBconcessionsBcommunicatorBcommunicateBcombinedBcombatBcoldB	cockerellBclosingBclickBclearedBclayBclauseBclashesBcircumstancesBchoseBchipsBchicagoBcheatsBchatBchasingBchartBcharltonBchannelsBchairsBceoBcelltickBcautiousB	catalogueBcatBcaribBcardiffBcapableBcaoBcairnBbwalyaBbuyerBburnerBburdenBbruceBbrokenBbroadcastingBbritBbristolBbrightonBbremenBbreastBbreakingB	brazilianBbrazilBbrandoBbottomB	borthwickBborderBboomB
bondarenkoBbodiesBblocksBblindBbizarreBbetBbergkampBbergerB
benefitingB	beautifulBbayerBbatterBbarsBbarryBbarrelBbarredBbarnBbalancedBbailBbafanaBbadlyBauthorsB	authorityBauctionB	attendingB
attendanceB
attemptingBathletesBaspiringBarvindBarmBarguingBargueBappropriatelyBapplicationsBappearsBapologyB	antivirusBanimalsBanimalBanalysisBamicusB	amendmentB	ambitionsB	aliadiereBalastairBairportsBaimsBagreesB
aggressiveBagesBageingBagedB
affordableB	admissionBadjustedBaddsB
activitiesBactionsBacademyBabsenceB911B90B68B404B400mB400B38B22B1995B1970sB1913B120B02B┬г35mB┬г1bnBzombicBzeppelinByoungestB	yesterdayByeahByardsBxpBxboxBwtaBwouldbeBworryBworriesB	workshareBwoodBwonderB	witnessedBwithprofitsBwithdrewB	withdrawnBwifeBwidowB
widespreadBwhiskyBwesternB	wellknownBweighedBweblogBwebbBwearableBwayneBwatchedBwarnockBwarmupBwardrobeBwalshB
walkingtonBwalkBwadisBvolumesBvoidBvitalBvisitsB	virtuallyB	villagersBviennaBvidoBveteransBveniceBvendorsBvaughanBvaticanBvaluableBusbasedBusbBurgingBupwardBunsureBunityBuniqueBunhappyBunderstandsBuncutBumpireBumbrellaB	typicallyBtwoyearBturnerBtuneBtumbledBtrulyBtruancyBtrophiesBtriumphBtrickyBtricksBtrickBtrialsBtreatedB	travelledBtrainedBtragicBtradersBtownBtomorrowBticketBthusBthrowBthreateningB
threatenedBthatcherBterryB	territoryBtensionBtenseBtechnicallyBtearsBteamedB	taxpayersBtariffsBswitchedBswedenB
suspicionsBsuspendBsuspectBsurvivedBsurroundingBsurgeBsurelyBsuppliesB	superheroBsunshineBsuddenB
subscriberB	submittedB	strugglesB
structuresBstrivingBstrictlyBstreamBstradeyBstoriesBstoneBstephenBstauntonBstatedBstartsBstarredBstandsBstanceBstagedBstableBspotlessBsplitsBspiteB	spiritualBspinB	spidermanB
spectatorsBspectacularB	speciallyBspeakersBsoonerBsongsB	somewhereBsomewhatBsolidBsoleBsoldiersBsocialBsoapBslowlyBslipBslightBskillBsizesBsilverBsignificantlyBsignalBsiemensBsiegeBshutBshoesBshockedBshinesBshellBshallBsevernBseventhseededB
settlementBsettledBsessionsBsessionBserversBsergeantBsensibleBselectedBseeksBseedsBsecuringBsecuresBsectionsBseatsBseagramBscreenedBscrappyB	scrappingBscrappedBscottishBscotchBscotBsaumarezBsaulnierBsatisfyBsandraBsamsungBsailingBryanBrusedskiBruralBrubbersBroyB	rotterdamBroryBroomBronanBrollBrobustBrobredoBrobotBringtoneBringsB	rightmoveBrightlyBrickBrhythmB
revolutionBreutersBreunitedB	retailersBresurrectedB	respondedBresortBresolvedBresignB	reshuffleBresearchersB
republicanBrepresentationB	reportersB
reportedlyB
repaymentsB	reopeningBrenewedBrenderedBremovedBremoteBremedyB
relentlessB
relativelyBrelatedB	regardingBrefusesBrefuseBrefugeesB	referringB	referenceBrecruitBrecognitionB	recogniseB	receptionB	realisticBrealiseB	reactionsBreachingBrbsBrapidBranksBramB	rajasthanBraidsBrahaBrackedBracesBqueuesBquestioningB	quarterlyBquarterfinalsB	qualitiesBqcaBqatarBpursueBpurelyBpulsesBpulledB
publishersB	provisionBprovidesB
prototypesB	protestedBprotestBprotectionsB	protectedBpropositionB	proposingBpromptlyB	promisingBpromisesBprofitabilityBprofileBprodigyB	procedureBprixB
prioritiesB	principleB	principalB	preventedBpreventBprestonB
presidencyBpreserveBpresenceBpreparationsB
preferenceBpracticeBpoweredBpoundB	positionsBportraitBportmanBportabilityBpoorlyBpoolBpollsBpledgedBplaystationBpingBpierceBpicturesBphysicsBphilipsBphilipBpestonB
persistentB	permittedB	permanentB
percentageB	pensionerBpendingBpeersBpearsonBpaymentBpatternBpatentsB
passengersBpartnersBpapersBpainkillersBowedBoverwhelmingB	overnightBoutfitBoutdoorB	otherwiseBoscarsBorqueraBopusBoptionsBopticBopponentBoperatorBoperatedBonsB	offensiveBoddsBoddB	occasionsBoccasionBnumerousBnudityBnoticesBnorwichBnonusBnitogliaBnickBnicholasBnicheBnewlookBnewerB	neverlandBnestorBnbcBnaziB
nationallyBnathanBnarrowlyBnarrowBmutantsBmusicianBmsnBmp3B	motivatedBmoscowBmorrisonBmonthlyBmonacoBmodestBmodernBmodeBminuteBmillerBmidlandsBmiddlesbroughBmichiganBmethodsBmethodB	messengerBmeritsB	mentionedB	mentalityB
membershipB	medallistBmcguireBmaybeBmatesBmassenetBmarvelBmarrBmarlinBmarketedBmarioBmappingBmapBmanuelBmannerBmalfunctionBmaleBmalcolmBmainsBmainlyB
madagascarB	lytteltonBlyingBlusakaBlureBluckyBlowestBloverBlovedBlouvreBlosesBlosB	looseheadB	loopholesBlongrunningBlockheedBlobbyBlleytonBlivedB
litigationB	literallyBlishmanBlinedBlindsayBlimBliftedBlicenceB
libertinesBlewseyBleroiBlegendBlegallyBleavesBlearningBleamyBleadsBlawnBlangdonBlandingBlandedBlandBlaidBladyB
laboratoryBkoreanBknewBkilroyBkidsBkickedBkhodorkovskyBkeynoteBkerrangBkennyBkeepsBkBjudicialBjudgmentBjoeBjimBjetBjacquesBjackmanBisnBipodsBipcBinvolvesBinvolveB
invitationB	invisibleB	inventionBinvasionB	introduceBinterventionBinteriorBintelligentBintegralBintactBinstructionsBinstitutionBinstantB
installingBinstallBinsistsBinseeB
initiativeBindustrialisedBindoorsB	indonesiaBindividualsB	indicatedB
indefiniteB	incumbentBimprovementsB	implementBimmenseB	immediateBignoredBignoreBidealB	hyderabadB	humphreysB	householdBhoulihanBhotBhostedB	hospitalsBhorribleB	hopefullyBhopefulBhonoursBhollowayBholidaysBholidayBhodgsonB	highspeedBhighprofileBherselfBheadlineB
hartlepoolBharrisBhardlineBhappensBhandingBhammerBhamBhaltedBhallBhalftimeBhailedBhaasBhBgunnersBguardB
guantanamoB	gronkjaerBgreyB	graphicalBgraemeBgovernmentsB	goldeneyeBgimmeBgillingwaterBgillBgiggsBgigBgethinBgeorgewbushcomBgeoffreyBgeoffBgenuineBgenreBgdpBgatherBgatesBgapBgameplayB	gallacherBgainedBfuturesBfulltimeBfuelledBfrustratingBfranticB	franchiseBfowlerBforthcomingBformingBformatsBformatBformalBfootageBfoleyBfoggBfloodingBflawBflairBflagBfischerB	firstteamBfirstlyB	firewallsBfindingsB	financingBfinalsBfilmingBfilingBfiledB	fightstarBfierceBfibreB	fernsehenBfeeB
federationB	featuringBfearedBfaxBfatboyBfameBfallenBfalconerBfailsB
fahrenheitBfactorBextraordinaryBextendedB
exploitingBexperiencesBexperiencedBexpandBexoskeletonsBexistB
exhibitionBexerciseBexcuseB	exclusionBexceptionalBethiopiaBeternalBerewashB
equivalentBequalityBepisodeB
envisionalBentirelyBenteringBensuredBenronBenjoysB	enjoyableBenjoyBengineBengerwitzdorfBenemiesB	endlesslyBenablesB
employmentBempireB	emotionalBemailsBelenaBelementsB	electronsB
electorateB	electoralBeighthBegyptBeffortsB
efficiencyBearnsB	earmarkedBdryBdruryBdrewBdrawingBdraftingBdraftedB	draconianB	downloadsBdouglasBdoubtsBdoublefaultsBdotBdoomBdominicaBdogsBdoctorsBdjBdividedB	diversityB	diversifyBditchB
distributeBdistanceBdisneyBdisguiseB	discussedBdiscoverBdisagreeBdireBdieBdicaprioBdevelopmentsB
determinedB	destroyedBdesktopBderbyBdepthB	depressedBdeportedB	dependingB	departureBdenyBdenounceBdenisBdempseyBdemonstratingBdemonstratedBdellBdelayedBdeiB
definitionB	defendingBdefeatedBdefaultBdeeplyB	dedicatedBdechyBdebitBdebatedBdealingBdaviesBdarfurBdantonB	dangerousBdanBdamagingBdamageB
culminatedBcrowdsB	cristianoBcreationBcrazyBcraigBcoxBcoverageBcountrysideBcouncilsBcorrespondentB
correctionB	convertedB
conversionB
controlledBcontestantsB
containingB
consultantBconsultBconstitutionalB
conspiracyBconnorBconnachtBconflictBconfidedB	conductorBconcertsBcompromisedBcomplexB	complaintBcomplainingBcomplainB	competentBcompactBcommonwealthBcommitmentsBcommissionedBcomfortableBcomedianBcolumnBcohenBcoachedBcloudBclosureB	clinchingBclimbingBcleverlyBcleverBclassicBclaimingBcitingBcirculatingBcinemaB	christianBchristBcherninB	chepkemeiBcheckedBchartsBcharlieB
charactersBchaosBchairedBchairBchainBcenturyB	celebrityBcelebrationBcbiB	cautionedBcatholicBcarmakerBcaribsB	caribbeanB	captaincyB
canterburyB	cannibalsBcannibalismB	campaignsBcampaignersBcamerasB	callaghanBcablesBcBbyrneB
butraguenoBbutlerBbusyBbustedB
bushcheneyBbuntesBbuiltinB	buildingsBbuenosBbudgetsBbudBbrothersBbrotherBbrosnanB	brooksideB
broadreachBbroadenB
broadcastsB	brilliantB	brightestBbriberyBboweBbosveltBborrowBboroBboomingBbondiBboltonBbobBblockbusterBblamingB	blackburnBbitsBbillionaireBbernabeuBbeliefBbayBbassistBbarriersBbarBbankingB	bandwidthBbalanceBbaikalBbabyshamblesBavoidingB	automaticB
authorisedBauditorsB	attractedB
atmosphereB	athleticsBastonBasiaBashleyBashamedBartBarseneBarrayBarchitectureB	approvalsBapplyBappealsBantonioBantiterrorismB
antipiracyB	antifraudBanthonyBanniversaryBanneBangerBangelesBanelkaBamplifyBambitionBambaniBamazonBamazingBalwaysonBalvaBalmuniaBallpartyB	allegedlyBallegationsBalevelsBakaBairlinesBairesB
agreementsBaffordBaffairB	advocatesBadultsBadultBadoptBadamBactiveBactingB	achievingBachievementBaccurateBaccompaniedBaccidentBabusedB	abilitiesBabcBabandonmentB86B800mB700B67B65B57B550B53B42B3dB34B	31yearoldB	30yearoldB30mB2dB	28yearoldB	25yearoldB20bnB2010B1996B1993B1991B1990B1920B1812B178B170B149B11mB11bnB107bnB07B┬г99B┬г8B┬г7960B┬г750B┬г6bnB┬г558mB┬г399B┬г374mB┬г35bnB┬г35B┬г2bnB┬г2932mB┬г25bnB┬г21mB┬г21bnB┬г200mB┬г20B┬г2B┬г10545B┬г1BzurichBzoeBzimbabweBzenB	zellwegerBzeeuwByugoslavianbornB
youngstersByakubuBxvBwronglyBwreckB
worthwhileBworstBworsleyBworryingBwormsBworldsBworkloadBworkerBwooingBwoodhillBwonderstrikeB	wonderingBwoesBwinsletBwifiBwiderBwhomB	wholesaleBwhoeverB	whicheverBwhereasBwestonB	westfalenBwerenBwellsB
wellplacedBwelcomedBweiseBweirdBweighingBweeklyBwearingBwearBweaknessBweakestBwavesBwattsBwatkinsBwashedBwarnsBwarnB	warehouseBwardBwalksBwalkedBwaldenBwagesBvotesB	voldemortBvolatileBvoipBvoigtB
vocationalBvivancosBvisualsBvirginBviolatedBviolateBvindictivenessBvincentBvillageB
vigorouslyB	vigilanteBviewingBviewedBviedB	viabilityBversusBvergeBvaryBvarietyBvansBvandalsBvaidyanathanBvacantBusledBupwardsB	upsettingBupsetBuprightBupperBuploadBupgradesBupdateB	unwillingBunwellBunveilsBunsuccessfullyBunsolicitedB	unpatchedBunfortunateBunfairB
unemployedBunderstandablyB
underlyingB
undergoingBundergoBunderexposureBunder21BuncoverBunclearBultimateBubisoftBtypicalBtypesBtynesideBtycoonB	twothirdsBtwodayBtuppenceBtulipBtrojansB	triumphedBtrillionB	triggeredBtriggerBtributeB	triallingBtrevorBtreatBtraviataB
travellingBtrappedBtransparentBtransparencyBtransmissionB
translatedBtransactionBtrainBtrailsBtrailB
tragicallyBtraffordB	trafalgarB	traditionBtradedBtractorBtraceBtouristBtourismBtouredBtoulouseBtoughestBtougherBtouchyfeelyBtouchingB	tottenhamB	totallingBtopplingB	toleranceB
titterrellBtiredBtinkerBtimingBtigerBtierB
ticketlessB	thresholdB	threetestBthreequartersBthreatenBthousandB
thoroughlyBthomsonBthompsonBtheoB
thankfullyB
terroristsB	terrifiedBterrificBtendonBtenderBtendBtemptB	temporaryBtemperaturesB	televisedBtelephoningBtelB
techniquesBtaxpayerBtaxationB	targetingBtamaBtalentsBtaleBtakeupBtakeoffBtaipeiBtaggingBtacticBtackledBtabloidBsymptomsBsympatheticallyBswingBswiftBsweptBsweepsB	swazilandBswappingBswapBsvanbergBsuzukiBsustainableB
suspensionBsusanBsurviveBsurveysBsurveyedBsurveillanceBsurprisinglyBsurpriseBsupremeB
supposedlyB	supportedB
supergroupBsungBsuitsBsuitorsBsuiteBsuitableBsuitBsuisseBsuingBsudaneseBsuccessfullyBsubtleB
substancesB	subsidiesBsubsidiariesBsubmitB
submissionBstylishBstuffedBstudiedBstudentBstubbornBstruggleBstrikersB	strihavkaBstrictB	stretchedBstressBstrengtheningBstrengthenedBstreamsB	streamingBstratocasterBstrathclydeB	strangersBstrangeBstrainedBstormedBstockingB	stipulateBstewartBstevieBsteroidsBsteroidBsteppingBstemBsteelBstaysBstayedBstaterunB
stateownedBstarfoxBstanleyBstanB
stagnationBstabbedBsquibsBsqueezedBsqueezeBsquarepantsBsquareBspurBspringboardBsprainBspottedBspotsB	spongebobBspokespersonBspiritsBspectrumB	spectatorB
specialiseBsparkedBspareBspammingtypeBspammerBsourcesBsoundsBsoughtBsortedBsophisticatedB	sometimesBsolvedBsoloBsoftenB	societiesBsnecmaBsmoothB
smatteringBsmashBsluggishBslowingBslowerBslowdownBsloveniaBslotsBslotBslippingBslimBslideBslamsBskisBskirtingBskinBskilledBsketchB
sixyearoldBsixweekBsittingBsisterBsinkingBsingledBsingersBsimplifyBsignalsBsightsBsidescrollingB	sibierskiBshyBshrugsBshruggedB	showcasedBshoutingBshouldnBshoulderBshotsBshortsB
shortrangeBshortlyBshopBshockingBshipsideBshippedBshiftBsherryBsheridanBshepherdsonBshearerBsheahanBsheaBsharplyBshareholderB	sharapovaBshakespeareBshakeBsexBseverelyBsettingsBservingBserialB
separatelyB	sentencedBsendersBselloutB
selftitledBselfemployedBselfconfidenceBselfBseizeBseededBsecretBsecondstringBsecondlyB
seasonallyBsearchedBseanBsealingBsealBseaBscrummagingBscriptBscreensBscoringBscorerBscoopedB	schweppesB	schroederBscholarshipBschnyderB	scepticalBscentBscenesBscenarioBscathesBscaleBsavingBsaversBsapporoBsantryBsansBsanctionBsamprasBsaluteBsalaryBsalariesBsaintBsailedBsadlerBrushedBrunnerBrungBrumouredBrumBruinedBroyceBrowntreeBrowingBrouteBrotateBrosneftB	rosenborgBronaldBromenBrolledBrolesBrolandBrogueBrodrigoBrodBrockersBrobbieBroadsBriverBrivalryBricheyBreyesB	rewritingBrewardedBrewardBrevivalB	reviewingBreversedBreverseBrevealBrevampBretroBretireBretainBresumedBrestoreBresponsetimeB	responsesB
respondingBresolutionsBresistedB	residentsBreserveB
researchedBrepresentingBrepresentativeBreporterBrepliesBrepliedBreplacesB
repeatedlyBrepeatedB	repealingBrentBrenownedB	renovatedBrennardBreneeBremotelyBrememberBrelievedBreleasesB	relationsBrelationBrejoinB
reiteratedB	reinforceBregulationsBregrettablyBregistrationBregisterBregionsBregardBrefusalBrefugeeBreflectsB	reflectedBrefiningB
refereeingBrefereedBreferBreevesB	reemergedB	redraftedBredraftBrediscoveredB	recordingBrecorderBrecordbreakingBreconstructionBreconstructedB	recessionB	receivingBreceivesBrebutsBrebuildsBrebuildBreassureBreapBrealisticallyBreactedBreachesBrayBraveBraucousBratsBratoBrarityBrarerBraphaelBrapBrankingsBrangingBrangesBrangersBrallyBrainingBrainBracingBracialBracedBquotedBquietBquestBquartersBquarterfinalB	qualifierBqualificationBquakingBqprBpursuedBpurposesB	purchasesBpunishedB	publicityBpsvBproxyBprowessBprovisionallyB
provincialBprovinceB
protectingBprospectiveBprosecutionB
prosecutedBproposeBproportionateB
proportionB	promptingB	prolongedBprogrammersB	professorB	producingB	processedBproceedsB
proceedingBprivatisationBprisonsBprisonerBpriorityBpriorBprinzBprintB
principlesBprimusBpriestB
prettejohnB	pressuresB
presidentsB
presentingBpreparesBprepareB
preorderedBpremiumBpremiseBpreferencesBpreferBpredictionsBpredictBpredecessorBpreconditionsB
precedenceB	pragmaticBprBpoweringBpountneyBpoundsBpotentB	postponedBposterB	postenronBpossibilitiesB
possessionBportugalBportsBpornBpoorestBpongBpollingBpointingBplotBpleaseB
plantationBplanksBplaneBplaidBpitchedBpiratedBpinsentBpieceBpickingsBpianoB
photographBphotoBphilippoussisBphilippeBpheasantBpfizerBpetrinaBpesoB	persuadedBperspectiveB
personallyBpermanentlyBperceptionsB
perceptionBpegBpeaksBpaysBpayoutBpayneBpattyBpattisonBpatrolsBpatrolBpatrickB	patentingBpassageBpassableBpartnershipsB	partneredB	partiallyBparodyBparmarBparmaB	parentingBparentB	paramountBpaoloBpalookavilleBpaintedBpainfulBpagesBpactBp3BoxfamBowusuabeyieBowningBownbrandB
overturnedBovershadowedB
overseeingBovercameBover55sB
outweighedBoutweighBoutsourcingBoutrightBoutingsBoutbreakBothelloBospreysBorionB	orchestraBoptimismBoptedB
oppressiveBopposeBopinionsBoperationalBopennessBopecBoneclickBomissionBolofBollyBoffshoreBofflineBofficersBoffencesBodayBoccasionallyBobtainedB	observersBobservationB	obscurityBobligationsB
obligationBobjectedBobeBnursesBnudeB	novelistsB
nottinghamBnotingBnotedB
nonfictionB
nominationBnominalBninthBniebaumBnicolasBnewscastersBnewlycrownedBnetherlandsBnerveBnemoBnegotiationsBneckB	necessityB	necessaryBnatureBnathalieBnatashaBnasdaqBnamibiaBnaBnB	mysteriesBmuslimBmusingsB	musiciansBmusampaBmurkyBmurderedBmultipleBmultibillionBmubangaBmrsB
mozambiqueBmouthBmountedBmottB
motorcycleB
motivationBmotionBmortonB	mortgagesBmorrisBmoralBmoodyBmolsirisBmobiliseB
missellingB
misgivingsBmisdirectedBmischiefBmiracleBminorBminisB	minibreakBmindsBmikhailB	migrationBmigrantBmidnightBmid2005B
microphoneBmickBmichelleBmetropolitanBmetresBmetadataBmessBmepsBmenziesBmentallyBmemoriesBmemorialBmelzerBmellbergB	melbourneBmehrtensBmefinB
medallistsBmeaningBmeadowsBmcmanusBmcdonaldBmbloxBmayorsBmayorBmaximiseBmaturityB
mattressesBmatrixBmathieuBmatchmakingBmasochisticBmarriageBmarlonBmarignyBmarianneBmarekBmardyBmantleBmanicsBmalawiBmajorityownedBmaintainingB
mainstreamBmahindraBmagpiesBmagnierBmaggsBmadameBmacraeBmacqueenB	macintoshBlyonBluringBluckilyBlowsBlowcostB	louisianaBloughboroughBlorryBlorriesBlopezBlongtimeBlongestBlodgeBlocksBlocationBlocallyBlobbiesBloathBloadBllanelliB	ljungbergBlividBliverpudliansBlitreBlitBlistedBlisbonBlisaBlipietzBlionB
linuxbasedBlimpedB	limelightBlimboB	likeliestBlifetimeBlifethreateningBlibelBliBlewisfrancisBlessonsBlesothoBlensBlengthyB	leibovichB
legitimateBlegalityBlegBlearnedBleapBleahyBleaflikeBlayingBlawsuitsBlauraB	launchingBlaunchesBlaughingBlathamBlastgaspB
largescaleBlargelyBlaptopsB
landsdowneBlandmarkB	landlinesBlanderB	lagardereBlacksBlackingBlackedBlabelledBkudrowBkudosBkronbergBkoreaBkodakB	knowledgeBkitBkieranBkeysBkerryBkentishBkenterisBkempBkeanuBkateB	kaplinskyBjunejulyBjumpingBjulianBjudgeledBjournalBjosephBjosepBjos233B	johanssonBjkBjetsBjeremieBjennieBjamaicaBjainBjadeBiwataBivBitv1B	isolationBisolatedBisleBislamBipv4BinvitedBinvestorBinvestecBintroductionBintimidatoryBinterviewedBintertwinedBinternationalsB	interfaceBinterestpaymentB	interceptB	intentionB	intensityBintendBintegrationB	insuranceB	insultingBinsultB
insulationBinstitutionsB	inspiringBinspiredB	inspectorBinsistB
innuendoesB
innovativeB	innocenceBinjuredB	injectionB	ingrainedBinformBinflationaryB
infightingB	inferenceBinfectsB	infectingB
inevitablyBinefficientB	inductionB
indonesianB
indicationBindicateBindiansBincurBincidentBinchesB	incentiveBincarnationB	inauguralBinadvertentlyBimproperBimprisonmentB
impossibleBimplyBimpliesBimplementedBimpetusBimpetuosityB	impendingBimmeltBimeldaBimagingBimageryBillnessBiiiBifsBideologyBidentityBidentifyB
identifiedBideallyBicmBicaBhytnerBhypeBhutchBhusseinBhuntersBhumorousBhumanitarianBhubsBhsdpaBhowlBhoweB
householdsBhoundsBhotelsBhospitalityBhorseB	horrifiedBhorrificBhoreBhoranBhopelessBhopefulsBhoopleBhoonBhomelandB
holdsworthBholderBhl2BhispanoBhintedB	hindustanBhijackBhighendBhideousBhesitateBherniaB
heptathlonBhensonBhelenaBheinzeBheatingBheatBheadsBheadquartersBheaderBhayesBhavanaBhaulageBhatBhasselBhartzBhartBharrisonBharmB
harlequinsBhappierBhansenBhanksBhangsBhandlingBhandlersBhandledBhandfulBhanauBhammeredBhammBhamiltonBhaltingBhaltBhaloBhalfwayBhalBhackersBhackBguscottBgunsB
guidelinesBguidedB
guaranteesB
guaranteedB	griffithsBgrievingBgreganBgreetedBgreetB	greenspanBgrantBgrabB	governingB
governanceBgoodnessBgoodbyeB
goldfingerBglobeBgloatingBgirlsBgigabyteBgielgudBgerhardBgereBgerdBgerardBgeorgiaB
geographicBgeniusB
generosityBgencrowdBgeldofB
gargantuanBgarethBgammellBgamblingBgambleBgainsBgainingBfullyfitBfuelsBfuelefficientBfrozenBfrontrunnersBfriendlyB
friendliesB	frequencyB	frenchmanBfreezingBfreemanBfreekickBfreedB	fracturedBfoxesBfoughtBforumsB	forresterBformersBformedBformallyBforestBforeBfooBfondBflowingBflowersBfloatedBflickerBflexibleBflexibilityBfledBflashBflaminiBfittingBfitsBfishBfirstsetB
firstroundBfirewireBfinlandBfinishesBfingerprintsB	filmmakerBfilledBfileBfightsBfightersBfiercelyBfieldedBfictionBfeudBfernandoB
fenerbahceBfenderBfemaleBfeedingBfbiBfaultsBfathersBfateBfascinationBfarmersBfantasyBfancyBfamineBfamilyfriendlyBfamedB	falsehoodBfalseBfallowBfalconsBfaheyB
facilitiesB
facetofaceBeyreBeyecatchingBextremeB	extensionB	extendingBexpressBexposedB	exportingBexportBexpoB
explosionsB	exploitedBexplainsB
explainingBexpiresBexperiencingBexpensesBexpandsBexmodelBexitB	existenceBexistedB
exemptionsBexcludeBexamBexactBeverettBevansBeurozoneBeuroscepticB	europeansB
euroafricaBethernetBestelleB	establishBessexBerrorBernstBeraseBequityBequallyBepicBenvironmentsB
entrylevelBentireBentersBenteredBenricoBenginesBengineeringBengineerBengagementsBenduringB	endorphinBendingB	encourageBencounteredBemployB	emphasiseBemphasisBemilioBemergingB	emergencyBembarrassmentBembarrassingBembarrassedBeliteBeligibleBelevatedBelectricityB
electricalBelectorsBelectioneeringBelB	eindhovenBegyptianB	eggshapedB	educatingBeditsBedgesBeddieBecbBeasterB
eastendersBeagerBdwainBduoBduncanBdubiousBdubbedBdualBdroppingBdriversBdriverBdrinksBdramaticallyBdragBdraculaBdozenBdoyleBdownsideBdownloadableBdowBdougBdoubtersBdoubledigitBdorivalBdoorBdoomedBdonmarB
dominationB
dominatingBdoctorowBdittoBdistressingBdistinBdissolveB
dissipatedBdiscriminationBdisciplinaryB	discardedBdisappearedBdisallowBdisadvantagedBdippedBdinghyBdimonBdigbyBdigB
difficultyBdifficultiesBdiesBdidierBdeteriorationB	detainingBdetainBdestructionBdestinationBdesperatelyB	despairedBdespairB	designingBdeservesBdeservedBdeserveBdescriptionB	describesB	descendedBdescendantsBderegistrationBdeprivedB
depressionBdepressBdeppBdepositsB	dependantBdependBdeniesBdenBdemiseBdemandsBdemandedBdeliveryB	delistingB
delegationBdelayingBdefineBdeficitsBdeferBdefeatsBdeclinesBdeclarationsBdecisionmakingBdecadesB
decadelongBdealtBdealingsBdeadlineBdeaconB	davydenkoBdaughterBdatabaseBdashingBdarkerBdannyBdanielaBdancingBdaimlerchryslerBczechBcyprusBcymruBcurveBcursingBcuriousBcurfewBcurepipeBculturalBcuetoBcubanBcryBcrossedBcroatianBcriticisingBcrimsonBcriminalityBcriedBcredibilityBcredentialsBcreatureBcreamBcrawlingBcrankBcraftedBcrackBcoveredBcourtsB	courteousBcoupledBcountyB
countrymenBcountBcottonBcostaBcosafaB
corruptionBcorruptBcorriganBcordBcorcosteguiBcoppinBcopperBcooperationB	cooperateBcooganB
convincingBconversationB
convenientBcontroversiallyBcontributorsBcontributingBcontributedBcontractualBcontinentalBcontentiousB	contenderBcontendBcontemporaryB	containedBcontactsBconsumerismB
consultingBconsultationBconsistentlyB	considineBconsiderationBconsciousnessB	connectedBconlethBcongregationBcongratulatedB
confoundedBconformB
confirmingB	confessesBconfederationB	conditionBconciliationBconchitaB
concessionBconcentratedBconcentrateBconceivableBconB
compromiseBcomplyB
completingB
competitveBcompetitorsB
competitorBcompetitivenessBcompetesBcompensationB
compatriotB
compatibleBcomparesBcommunitiesBcommuniqu233BcommunicationBcommonplaceB
committingB
commentingBcommandBcomicBcombineBcolouredBcolourBcolombiaBcollegeB	colleagueBcoincideBcoheadBcoffersBcoastalBcoachingBcoachesBcluesBclothingBclosetsBclockedBclinchB
clearswiftBclaxtonBclassedBclashingBclareBcivilianBcitizenshipBcitiesBcitedBcisseBciscoBcircuitBcinemasBchurchBchristopherBchipolopoloBchinB	childcareBchildB	chemistryBcheatingBcheaperB	charitiesBchaptersBchainsBcertificateBcernBcentsBcentralisedBcellBcelebrexB	celebrateBcdsBcavanaghBcausesBcauldronBcattB
categoriseBcategorisationB
categoriesBcastrogiovanniBcastleBcashrichBcarriersBcarpetBcarolinaBcarmanBcarefulBcaptBcapsBcapitalisationsBcapitalisationB
capacitiesBcapabilitiesBcapBcannesB	cancelledBcancelBcameronBcameoBcambridgebasedBcallupsB
californiaBcaf233BcadburyBbyelectionsBbyeBbuzzBbuyersBbustBbusesBburstBburnleyB	burnerdvdBbureaucracyBburdenedBbuoyedBbumpBbulletinBbugsBbuerkBbudgeBbuddBbrutalBbrowsingBbronzeBbroadlyBbroaderBbrixtonBbritonBbrinkBbriefingBbriefB	brentfordBbremnerBbreezesBbreedBbreachesBbovinaBbournemouthBbourneBbotswanaBbotafogoBborrellBborisBboreBbootsBboormanB	bombardedBbombB	boixvivesB	blueprintB
blindinglyBblendingBblatterBblastsBblastedB	blanchettBblamedBblackmanBbiteBbirdBbilledB	bilateralBbigotryBbigleyBbiddingBbibleBbetisBbestsB
berlusconiB
bergamascoBbennyBbeneathBbeltBbelongBbelmarshBbehindthescenesBbegBbeachBbbc2BbbaBbatesBbasfBbarrosoB
barristersB	barristerBbaronessBbarelyBbarbaraB	bannisterBbanningBbankruptBbankerB	bandwagonBbandsBballsBbaladoBbaggageBbaftaB	backwardsB
backtobackB
backgroundBawolBawaitBavivBavineriBaviatorBauthenticateBaustrianB	austerityB	audiencesBattractionsB
attentionsB	attentionB	attemptedBattackedBattachmentsBattachedBatpBatleticoBatlasBathlonB
asymmetricB
astiazaranBassumedB
associatedB	assistantB
assessmentBassessBaspectBasksBashfieldBashesBarturoBartisticBartistB
artificialBarrivalBarrestsBarrestedBarrangedBarmsB	armisteadBarmandBarisenBarguablyB
argentinesBarenB	architectB	arbitraryBarbitrarilyBarabBapproveBappropriateB
approachedBappreciatedBappointBappliesBappliedBappealedBaonBaolBanywhereBanywayBantispammerBanticorruptionBantibushBanswersB	anonymousB	announcesBanniesBannaB	animationBangleBangelB	ancestorsB	anastasiaBanabolicB	amsterdamB	amplifiedBamountsB	amountingBamountedBamongstBamendingBamdB	ambitiousB	amazoncomBamazedBamandaBalunBaloudBaloneBallegingBallawiBalertedBalertBalbeitBalarmBalainB	aizlewoodBaiyarBairlineBairingBaidsBaidingBaiBagmBagentsBafricanamericanBafghanistanBaffectBadvisoryBadvisorsBadviserBadvisedBadviseBadvertisingBadslBadoptionBadoptingB
admirationBadministratorB
adequatelyB
adaptationBadairBactualB	activistsBacrimoniousBacquireBachillesBaccusingBaccusationsBaccuracyB
accountingB	accountedBaccountancyBaccountabilityBaccomplishedB
accomodateB	acclaimedB	accessingB
accessibleBacceleratingBabusesBabbottBabbasiBaaraBa380B98B96B90sB900B8742mB87B850B85B84B83B800B78B73B700mB68mB66B659B5mB5iveB56B54thB5498mB54B51B50thB50mB5050B48B41B407thB39mB39B37B33B30manB301bnB300shareholderB29aB289B281B	26yearoldB26bnB239bnB23B	22yearoldB2200B	21yearoldB21mB206B200mB200506B19bnB1990sB1989B1987B1985B1984B1960sB1949B1930sB192pB187thrankedB1865B184B1831B1800sB	17yearoldB175B16bnB15mB14bnB141bnB1400B138B136mB136069B133bnB130B12bnB128B11thB10yearB10thB10mB108B1038amB100bnB06B05B030B007B000mB┬г8mB┬г8bnB┬г885mB┬г85bnB┬г857mB┬г808mB┬г800mB┬г7mB┬г799B┬г72bnB┬г723mB┬г722mB┬г71bnB┬г707mB┬г686mB┬г644mB┬г600B┬г6B┬г5mB┬г581mB┬г57bnB┬г55bnB┬г550mB┬г53mB┬г52mB┬г529mB┬г4mB┬г48bnB┬г469B┬г468mB┬г46B┬г450B┬г44bnB┬г429mB┬г400mB┬г400B┬г40B┬г4B┬г3bnB┬г385mB┬г375bnB┬г357mB┬г339B┬г31bnB┬г306mB┬г306B┬г298bnB┬г293B┬г29B┬г28bnB┬г287mB┬г286bnB┬г285mB┬г27mB┬г269bnB┬г268bnB┬г266B┬г255bnB┬г254bnB┬г24mB┬г23B┬г22mB┬г190B┬г19B┬г18bnB┬г189B┬г188bnB┬г17mB┬г170bnB┬г17B┬г162mB┬г16B┬г15bnB┬г157mB┬г157bnB┬г152B┬г14mB┬г138mB┬г130B┬г125bnB┬г11bnB┬г117mB┬г111mB┬г10bnB┬г108bnB┬г105mB┬г105bnB┬г102mB┬г10280B┬г10BzonesBzombiesleapingoutofshadowsBzombiesBziyiBzidaneBzhengBzhangBzensBzedBzanderBzambiansBzadieByusufByoursB	youngsterB	yorkshireByorkerByobBymByleByeoByellowByellingByearoldByearlongB
yeahronimoByaronByangByamaroneByBxrayBxp2BxmenBxfactorBxeniaBx800BxBwynBwsjBwruBwrongsB
wrongdoingBwriterdirectorBwrayBwrapBwowedBwoundedBworthyB	worthlessB
worldclassB	workspaceBworkrateBworkmanlikeB	workhorseBworkableBworcestershireB	worcesterBwoolwichBwoodheadBwoodenspoonB	wonderfulBwonderedBwolfgangBwokeBwojahnBwoefullyBwltsBwltmBwizardryBwizardBwivesBwithholdingB
withdrawalBwithdrawB
witchcraftBwitchBwishfulBwiselyBwiseBwiredBwireBwingsBwineBwindyB	windsweptBwindsorBwindBwillianBwillettsBwightBwidenB	wideangleBwhittledBwhistleBwhishawBwhipsBwhimBwhereverBwherebyB
wheelchairB
whatsoeverBwhaleB
wgbhbostonBwestcountryBwesBwelshmanBwellrecognisedBwellinsulatedB
wellearnedBwellcraftedBweingartnerBweightBweepingBweedBweddingBwebsidestoryBwebpagesBwebberBwebbasedBweaverBweaponryBwealthyB	weakeningBweakenedBweakenBwaysideBwayfarerBwavingBwatersupplyB	watershedBwaterconcessionBwaterandweathertightB	watchdogsBwastedB	wastealotBwassersteinBwassBwashingtonbasedBwashingBwashersBwashBwarwickBwarsBwarriorsBwarningsBwarmBwaresBwappushB
wandsworthB	wanderingBwaltersB
wallpapersB	wallowingBwallaceBwalkmanBwalkingBwakeupBwaivingBwagerBwadeBw800BvyingBvulnerabilitiesBvoterBvotecatchingBvolvoB	voluntaryBvoluntarilyBvoltageBvolleysB
volkswagenB
volatilityB	voiceoverBvoicedBvodkaB
vociferousBvivreBvividBvivendiBvisuallyBvisibleBvirginiaBvioxxrelatedBviolentB	violationB	violatingBvincenzoBvimaBvillainsBvillagesBviktorBvigorousBviewsBvidukaBvideosB	victoriesBvictimBvicechairmanBviceBvicBvianelloBviableBvhsBvetoB	verwaayenBverticalBvermontBverdictBverdiBverballyBverbalBveracruzBventuresBveneerB	vehicularB
vehementlyBvcrBvaulterBvatB	varietiesBvanishedBvanishBvaluingBvaluesBvaluedB
valueaddedB	valuationBvalleyBvaliumB
validationBvaleryBv20ButurnButterlyB	utilisingButahBustaBuserfriendlyBusageBuruguayBurgentlyBurgentBurgencyBurgeBurbinoBupweBupturnBuptonBuptakeBupswingBupsurgeBupstartBupstairsBupsetsB	uploadingBuplightsBuphillBupheldBupheavalB	upgradingBupgradeableBupdatesBupdatedBupcomingB	unwindingBunwillingnessBunusableBunsungB
unsettlingB	unsettledBunseasonablyBunrulyBunreturnableBunrestBunrelentingB	unrelatedBunravellingBunravelBunprotectedBunprofessionalB
unplayableBunpaidBunosatB
unofficialBunnamedBunlawfulBunkemptBuniversitiesBuniverseBunitaryBunisonBuniquelyBuninspiringBunifiedB
unfeasiblyB
unexpectedBuneditedB	unearthedBundoB
undertakenB
understoodBunderstatedBunderplayingBunderpinnedBunderperformBunderestimatingBunder19B	undecidedB
uncoveringBuncomfortableBuncharacteristicBuncertaintyB
unblightedBunbelievablyBunavailableBunassailableB	unanimousB
unadjustedBumbrageBultraB	ukrainianBukraineBukbasedBubiquitouslyBu2BtyresBtycoB	twoleggedBtwohourBtwohorseBtwitchesBtwistedBtwistBtwinsBtwinkleB
twicetakenBtwelveB
tvondemandB	tvenabledBtussleBtussaudsBturnbullBturkmenistanBturinBturfBtunnelBtunisianBtunisiaBtunesBtunedBtubB
tsunamihitBtshirtsB
tryscoringB
tryscorersBtrussesBtrumpedBtrucksB
trowbridgeBtroughsBtroughBtroublespotBtroubleshootingBtronconBtroikaB
triumphingB
trinationsB	trimmingsBtrimmingBtrimmedBtrimB	tricklingBtrekBtreatingB	treasuresBtravisBtravelsB
travelodgeB
travellersBtrapBtransportingBtransportationB
transplantBtransparentlyBtransmittedBtransmitBtransformedBtransatlanticBtransactionsBtrainsBtrainersBtrailingBtrailedB	tragediesBtraffickingB
traditionsBtraditionallyBtradedrivenBtracyBtracksBtrackingBtraceyBtracesBtracedBtowersBtowerBtoutedBtoursBtouristsBtouringB	toughenedBtouchscreenB	touchlineBtouchedB	touchdownBtossedBtorrentBtornBtorenBtoreB
topsellingBtopsBtoppledBtoppingBtopnotchBtopfiveBtopclassBtootweeB	toothlessBtoothBtoonBtonnaBtonlineBtonightBtoniBtongaBtoneBtolstoyBtollB	toleratesB	tolerableBtoiletBtobaccoBtivotypeBtittleBtitanicBtiscaliBtiruneshBtiresomeBtipsBtipBtinaB
timetablesB	timetableBtimeshiftingBtimehonouredB
timberlakeBtillBtightlippedBtighterB	tiebreaksBtideBtickingB	ticketingBtiagoBthwartsB
thunderousBthunderbirdsBthumpedBthumbsuckerBthumbBthuggishBthrownBthriveB
thrillrideBthrewB
thresholdsB	threeyearB
threejudgeBthreecorneredBthreeandahalfB	threatensBthoughtthroughBthoughtsBthoroughBthirdsetB
thirdroundB
thirdpartyB	thinktankBthinkerBthinBthgBtheronBtheresaB	thereforeBtherapyB	therapiesBtheirsBthefirstBtheaterBthatcheritesBthaliandBthaisBthailandBtexturesBtextilesBtexanBtetrahydrocannabinolB	testimonyBtestifyB	testamentBtescoBterribleBterrellB	terminalsBterminalBtepidBtenyearBtenfoldBtendsB	tenderingBtendencyB
tendenciesBtendedBtemptedBtempoBtempestBtemperatureBtemperamentalBtemperB	telephotoB
telephonedBtelecommunicationBtelecastBteethB	teeteringB
teessidersB	teenagersBteddyB	techsavvyBtechnophobesBtechnologicalB	techniqueBtechniciansBtebbitBteasedBtearfulB	teammatesBteamingB	teachingsBteachingBteachesBtcpipB
taxraisingBtaxingBtautouBtauntsBtaughtBtattonBtattleBtateBtasksBtappingBtapeBtantalisingBtanksBtandemBtampaBtamelyBtallyBtalkedBtakehomeBtagsBtagBtacticalBtacklecountBtabletBsynopsisB
synonymousBsyncingB
sympathiseBsymbolsBsymbologistB
symbolisesBsymbolicallyBsylviaBsylvainBsybariBswungBswellBsweetmanBsweepingBsweepBswedeBswearBswatheBswappedBswampingBswampB
sustainingB
suspiciousBsusannaBsurvivorBsurvivesBsurvivalB	surroundsBsurroundingsBsurreyBsurrenderingBsurrenderedB	surpassedBsurfwearBsurfingBsurfersBsurfacesBsurfBsupremoBsupposedBsupposeBsupportsB	supporterB	suppliersB
supplementB	supersizeBsupermarketB
superjumboBsuperintendentB
superclubsB	superbowlBsunkBsumsBsumitomoBsulpiceBsuitedBsuicidalBsugiyamaBsufficientlyB	sufferersBsuesBsudanBsuckBsucessorB	successorB
succeedingBsubtletyBsubstitutedBsubsidyBsubsidisingBsubsideBsubsequentlyBsubscriptionsB
submarinesB
subliminalBsublimeB	subjectedBsubirBsubconsciouslyBstupidBstuntBstunningBstumbledBstumbleBstudyingBstuckBstubbsBstruttedB
structuralB	strollingBstrokesBstringBstrikeBstrideB
stretchingB	stressingB
strengthenBstreepB
streamlineBstreamedBstreakBstraussB	strategicB	strangelyBstrandedBstrainBstraighttalkerBstosurBstormsBstoresBstoppingBstoppedBstomachBstolichnayaBstokeontrentBstokeBstocktononteesBstocksB	stockholmBstirBstifleBsterlingBstereotypesB
stereotypeB
stepchangeB
stepbystepBstefanBsteerB	steamtypeBsteamsBstealthBsteadyBsteadB	statutoryBstatuteBstatureB	statisticB	stationedBstaticB
statementsB	stategistBstatB	startlingBstaringBstanzelBstandupB
standstillB
standpointBstandoutBstandoffBstandinBstampingBstampBstalkerBstairsB
staircasesBstagnateB
staggeringBstadiumsBstadiumconqueringBstadeBstackedBstabilisingBsriBsquidBsquashedBspyBspurredBspudsBspritesB	sprintersBsprintedBsprintB	spreadingBsprayB	spotlightB	sportsmanBspoofBspontaneousBsponsorsBspokeB	splittingBspitzerBspiritedBspiritBspikeB	spielbergBspicejetBsphereBspencerBspellBspeedyBspecifyB	specificsBspecificallyBspecialisedBspeaksBspawnBspatesBspatBsparksB	sparklingBsparkBspamhausB
soundtrackBsoundingBsoulmateBsoulmanBsosoBsortingBsorrowBsophisticationBsoniaBsomehowBsolverBsolveB	solutionsBsolariBsolBsoilBsociededBsociedadBsoaringBsoandsoBsnubbedBsnubBsnowmenB	snowfallsB
snowboardsBsnowBsniffedBsnapsBsnappedBsnapBsnackBsmoothtalkingBsmokeBsmashingB
smartcardsB	smartcardBsmallbudgetBsmacksBslumpedBslumpBslowerthanexpectedBslowedB	slowdownsB	slovakianBslogansBsloganBslinkierBslickBslicedBsliceB	sleeplessBsledgehammerBslaterBslatedBslashBslamdunkBslackBslabB	skywalkerBskyBskirmishBskipBskimakerBskillsetB	skeletonsBskeletonBskBsizedBsixthminuteB	sixthformBsituatedBsitedBsinnersBsinisterB
singletonsBsinglepersonBsingingBsinghBsingerguitaristBsingaporelistedBsingaporebasedB	sinbinnedBsimultaneouslyB	simulatesBsimulateBsimsBsimplestBsimilarlytitledB	similarlyBsilvioBsillyBsilkBsignsBsigningsB	sidelinedBsidefootingBsicBsiberiaBsiacBshumB
shrewsburyBshrankBshowerBshowcaseBshoutedB
shotmakingB
shortlivedBshortlistedBshorterB	shortenedBshortageBshoreBshoppersBshooterB	shootemupBshootanythingthatmovesBshootBshocksB	shiveringBshirtBshirleyBshipsBshiningBshineBshiftsBshieldsB
shevchenkoBsheppeyBshepherdBshelvesBshelterBshelleyBsheilaBsheetBsheermanBsheerBsheddingBshawBshatwanBsharonBsharkBsharingBsharersBshareholdingBsharedBshankarBshamiBshamefulBshamedBshameBshamblesB
shailendraBshadowsBsfarBsexyBsexuallyBsexualB	severanceBseussBsetbackBsertitBserbiaB	sequencesBsequelsBseppB	separatedBseoulB	sentimentB	sentencesBsensingBsendsBsenderosBsempronBseminalBsemiconductorBsemelBsellsBsellerBselimaBselfpublicistsBselfishBselfharmBselffinancedBselfdiagnosisBselfdeclaredB
selfcensorB
selectionsB	selectingBselectBseldomBseizingBsegaBseesawB	secretiveBsecondgenerationBsecondfastestB	sebastienBseasonedBseasonallyadjustedBseasonalityBseasonalBseasideB	searchingBseamlessB	seachangeB	scupperedBscupperBscrubbedBscrubBscrollBscriptedBscrewedBscreenwriterBscreensaversBscreenfillingBscreamedBscrapsBscrapBscotsBscorseseBscootedBscoopsB	sclerosisBscissorsBscintillatingB
scientistsB	scientistB
scientificB	schwimmerBschv770BschumerBschoolchildrenBschmalerB	schedulesBscaryBscarlettBscaresBscaredBscarceBscarBscantBscansBscannellB
scandalhitBscamsBscamBscalaBsayleBsaviourBsavagesBsatoruB
satisfyingBsatifexB
satellitesBsarsBsarcasticallyBsapBsaoBsantoroBsantiagoBsantaBsandyBsandsB	sanctionsBsanctioningBsamuraiBsamanthaBsamBsakhalinBsakeBsailBsafelyB
safeguardsBsadnessBsadB	sacrificeBsackedBsackBsachsBsaabBsaBrupertBrunwaysBruncornBrumbledBruinsBrudolfBrudeBruckBrubensB	rubbishedBrubBrsaB	royaltiesBrowsBrowanBroutingBrouteraccessBrousedBroundedBroukisBroublesBrotatedBrotBrossiBrossBrosesBroostBrongB
ronaldinhoBronBromanticBrolloutBrokrBroiBrodneyBrockwoolBrocketBrockerBrobustlyBrobotsB	robertsonBrobertsBrobertoBrobertaBrobberyB	robberiesBrobbenBrobBroadshowBrksBriversB	riverbedsB
risktakingBriskedBrippedBripeBripBringBrigidB	righthandBrifeBridiculouslyB
ridiculousBrideBridBrichestBricherBricardoBricardBrhodriB
rhetoricalB	rewrittenBrewardsBrevolutionisingBrevokeB	revisionsBrevisedBreviewerBreviewedBrevertBreversesBrevengeB	revellersBrevelledBrevampedBreunificationB
retrievingBretrialBretrenchmentB
retrainingBretainsBretailerBresumeBrestsB	restoringB
restaurantBrestartB
responsiveBrespondentsBrespiteB	respectedBrespectableBrespectBresourceB	resortingB
resolutionBresignationsB	residencyB	reservoirBresentBresellBreseachBrescuesBrerunBrequiresBrequirementB
reputationBrepeatBrepayBreparationsBrepairsBrepairBrennickBrenegotiateBrenegadeBrenaissanceBremovingBremovalBremindedBremedialBremarkB	remainderB	reluctantB
reluctanceBreligionBrelieveBreliedBreliantB
relegationB	relegatedB	releasingBrelayBrelaxedB
relaxationBrelaxBrelaunchBrelatingBrelateBrejoinedBrejectsB	reiterateBreinvestBreintroductionB
reinforcedBreinBreidB
rehearsalsB
regulationB	registersB
regenerateBregardsB
regardlessBregardedBregalB	regainingBregainedBregainBrefutedBrefusingB	refocusedBreflexesBreflexB
reflectingB
refineriesBrefinedB
referencesBrefBreevaluationBreesB
reengagingBreeledB	reductionBredfordB	recyclingB	rectifiedBrecruitmentB
recruitingB	recruitedB
recordingsB	recordersBrecordbustingB
recordableB
reconvenedB	reconveneBreconstructB
reconsiderB
recommendsBrecommendationB	recommendBrecognisingB
recognisesB
recognisedBrecognisableB	recliningB	reckoningB	rechargedBrecessB	receptiveBrecentlycrownedBreceiptsB	recallingBrecallBrebuttalB
reboundingBreboundBrebirthB	rebiddingBrebelsBreassuranceB
reasonableBrearrangingBrealtiesBreallifeBrealityB	realisingBrealisedBreadilyBreaderBreactionBreactingBreactBraymondBraymanB	ravenhillBravagedBraulBrattlingBratingsBratifiedBratesettingB
ratchetingBratBrashlyBrashBrareBrapperBrappedBrapidlyBrankingBrangBrampantBrambertBrallyingBralliesBrainyBraininterruptedBrainerBrainbowcolouredBrailBragingB	radicallyBradeonBrackingBracketBracistBrachelBrabbitBrabahBquoteBquizzedBquittingBquirkyBquinnBquieterBquickerBquellBqueensBquashieBquashesBquarryB	quadrupleBqrioBpwcBpuzzlesBpurvinBpursuitsBpurposeBpurportBpurgeB
purchasingBpunkBpunishmentsBpunditBpunchesBpunchBpumpsBpumpedB	pulsatingBpullsBpullingBpucciniB
publicisedBpryceB	proximityB	provokingBprovokedBprovisionalBprovenB	prototypeB	protocolsBprotestsB
protestersB
prostituteB
prosperityB	prospectsBprospectBprosecutingB	prosecuteBpropostitionBproposesBproperlyB	propelledBproofBpromptsBpromptedB	promotionB	promotersBpromoteBprominentlyB	prominentBprojectionsB	projectedBprogressionB
progressedBprogrammingB
programmedB
profoundlyBprofoundB
profitableBprofilesBprofessionalsBprofessionalismBprofanitiesBproductionsBproddingB
processorsB
processingBproceedingsB
proceduresBproblematicBprivyBprivacyB	prisonersBprintsBprintedBprimaryB	primarilyBpricingBpriceyBpricewaterhousecoopersBpricedBprewarBpreviewsBpreviewB
preventionBprevaricatedBprevailBprettierB
pretendingBpretaxB
presumablyBprestigiousBpressingBpresidentialBpresentationBprescriptionBprescottBprerecordedB	preordersBpreorderBpremiereBpreliminaryB
prejudicesBpregnantB	preciselyBpreciousB	precedentBpreachesBpre11BprayBprattBpranabBpraisesBpraiseB	practisedBpractiseB	practicalBpozzebonBpowBpouredBpourBpostersB
postdotcomB
possibiltyB	positivesB
positionedBposedBposeBportugalbasedBportrayBportlyBportionBpornographicBpopeBpooleBpoliticallyB
politciansBpolishBpolicymakersBpolicyholdersBpolicyholderBpoleBpokingBpoisedBpognonBpoetryBpoetBpodsBpodcastsBpocketsBpocketBpobrezaBpngBpmB
plummetingB	plummetedBpluggedBplugBpledgingBpleasureBpleasingBpleasantBpleadBpleaB	playmakerB	plausibleBplathBplasterboardBplasterBplantsBplantedB
plantbasedBplantBplanetsBplanesBplainBplaguingBplacingB	placatoryBpitsBpistolsBpiratingBpirateBpippedBpioneersBpioneerBpinpointBpinksBpinkBpinBpilotBpilingBpileBpigeonBpiecesBpieB	pictochatBpiazzaB
physiciansBphraseBphotosBphotorealisticBphotonsB	photonicsBphotographsBphotographersBphonographicBphilharmonicBphasedBphaseBpharmaceuticalsBpewBpetrolBpetraBpetersenBpeteredB	pescatoreB
pervertingBperuginiBpersuasionsB
persuasionB	personnelBpersonalitiesBpersistsBpersicoBpersecutionBperreiraBperpetuatingBperpetuatedB
perniciousBpermitsB
permissionBperjuryBperjurerB
peripheralBperiodsB
performingB	performerBperformancewiseBperformanceenhancingB
perfectionBpercentB	perceivesB	perceivedBperceiveBpepperedBpeppeBpentagonBpennyBpennedB	peninsulaBpenetrationBpenceB
penalisingB	penalisedBpenBpelousBpeelBpedigreeBpedestrianisationBpectoralBpearlBpeakedBpdasBpcbasedBpayfordownloadBpayersBpavlikowskyBpavelBpauseBpauleBpattayaBpatrikBpatriciaBpatientBpatienceBpathBpatchedBpatBpasadenaBpartidoBparticleBparticipantsBpartialBparrBparisseBparibasBparanoidB	paparazziBpapBpanickyBpancakeBpalmedBpalancasB	paintingsBpaintBpainsBpaddyBpacmanBpacksBpacketBpackardBpackagesBpacingBp2pBozzyBoyensBoxfordshireBoxfordbasedBoxfordBoxbyBowainB	overwhelmBoverviewB
overvaluedBovertureB	overtonesB	overtakenB
overshadowBoverseesBoverseeBoverseasbasedBoverrunBoverrelianceBoverheadBoverestimatedBovercrowdingBoverconfidentB	overblownBover70sBover65sBovensBovationBoutsprintedB	outspokenB	outsourceBoutplaysB	outplayedBoutnumberedBoutmodedBoutlookBoutlinesBoutlineBoutlawB	outgunnedB	outgrowthBouterBoutdatedBoutcryBoutcomeBoutclassB	outandoutBoustedB	ourselvesBoursBoughtBotterB	ostrichesBoscarnominatedB	osbournesBosBorwellBorganismB	organiserBorchestrateB
orchestrasBorangeBoptinrealbigcomBoptBoppositeBopposesBopeningsB	openendedBopelBonwardBonuohaB	onthespotBonthegroundBontarioBonstartBonstageBonshoreBongaroBoneyearBonetooneBonetimeBonethirdBoneoffB	onelinersBonehundrethBonedimensionalB
oncefamousBonatoppBomittedBolympianBollieBolegB	oldschoolBoldsBoldestBokayBokBogB	offendersBoffenderBoffendedBoffenceBofcomBodeonBodedBoddjobBoddieB
occurrenceBoccurredB
occasionalBobtainbuildBobtainBobservesBobserveB
objectivesB	objectiveBobjectionableBobikweluBobeyBoasisBoakBo2BnyeBnuttersBnutsBnutBnurseryBnuisanceBnuclearBnowinBnowadaysBnovelsBnouBnottinghamshireB	notoriousBnotionBnotifiedBnotablyB	nostalgiaB	northwestBnorrellBnormanBnormallyBnormaBnontechnicalBnonseasonallyB	nonprofitBnongeographicalBnonessentialBnomineeB
nominatediBnoisyBnoisehitBnoiseBnofrillsBnoelBnodsBnoahBnirupamaBnikolayBnikeBnightsBnightmarishB	nightmareB	nightfireBnigglingBnigglesBnigerianBnigelBnielsennetratingsBnicoleBnicolBnickyBnickiBnicholsBnichollsBniaBnevisBneverthelessBneverendingBneutralBnetsB
netherdaleBnetbasedB	nervouslyBnellieB	neighbourBnegrasBnegotiatingB
negotiatedBnegativeB
needlesslyBneedlessBnearsBnearbyBneaptideBnaxosB
navigatingBnavigateBnationalityBnasaB	narrowingB	narrativeBnapsterBnapoleonBnapierBnamingBnamecheckedBnailedBnailBn64BmythB	mystifiedBmysteryBmutualBmutantBmusthaveBmusicenabledBmusicalsBmuseBmuscleBmurrayfieldBmurphyoBmurdochBmunozB	municipalBmumbaiBmumB	multitudeBmultitalentedB
multipolarBmultiplayerBmultinationalsBmultimillionBmultifunctionBmulticolouredB	mukherjeeBmukerjeeBmugsBmuddledB
muchneededBmtvBmtsenskBmountingBmountainB	mountableBmountBmouldedBmotorsB	motorheadBmotivesBmotivateB
moratoriumBmoraleboostingB
monumentalBmontyB
montenegroBmonsterinfestedBmonsterBmonotonyBmonopolyBmonicabasedBmondaysBmollycoddledBmolikBmogulBmoffatBmodifyBmoderateBmodelledBmoazzamBmiyazakiBmixtureBmixedBmixBmitsuiBmisusingBmisusedBmistranscriptionBmistakesBmississippiBmissionsBmissionBmisrepresentationB	misplacedBmisledB
misleadingB	miserableBmiscarriagesB
minoritiesBminnowsBminitournamentsBminitournamentB
ministriesBmingleB	minghellaBmineBmillsBmillarB
milestonesBmilesBmileageBmilanesiBmikoyangurevichBmightyBmidwayBmidtaskBmidstB	midmarketBmid2004BmidBmicrosoftpartneredBmicroBmichellBmichelBmiaBmi6BmiBmexicanBmeticulouslyBmeteoricBmessyBmerylBmerryBmerrillBmergingBmergersBmergeBmerelyBmereBmercyBmerchandiseBmenuBmentionsBmentionBmendietaBmendaciouslyBmenatepBmemoryB
memorandumBmemoirBmellonBmegsonB
megastoresB
megapixelsB	megabytesBmefBmediumB	medicinesB	medicallyB
mechanismsB	mechanismB	measuringBmearBmeantimeBmeandzoegateBmealsBmcmillanBmcilroyBmciloryBmcgregorBmcgeeB
mcewenkingBmcenteeBmcenroeB
mcculloughB	mccormickB	mccartneyBmccambridgeBmaxwellBmaxisBmaximovBmaxBmauriceBmaulB
matsushitaBmatronBmathsBmatchwinnerBmatadorBmastermindsB
mastermindBmasterBmassesBmasoBmasiBmartyrsBmartinaBmartaB
marshalledBmarriedBmarleneBmarksBmarkingBmarketplaceBmarkerBmaritimeBmarionB	marinuzziB
marginallyBmarginalBmargaretBmareBmarcoBmarcelloBmarcB	marathonsBmaratBmaracanaB
manuscriptBmanufacturersBmanufacturedBmanuallyBmanualBmanonBmanoeuvringBmannB
manmachineBmanjitBmanipulationB
manifestosBmaniBmangedBmangaB	maneuversBmandulaB	mandelsonBmandateBmanchesterbasedBmanagerofthemonthB
managerialBmanagementchicagoBmalnourishmentBmalloyBmalibuBmaleevaBmakeupBmakeleleBmaitlandB	maintainsB
maintainedBmainlineBmailsBmaidenBmagnusBmagicalBmaggieB
magdaleenaB	magazinesBmaduakaBmadonnaBmadnessBmadBmacsBmackenBmackelBmacbethBmaBlyricsBlyricistBlynneBlynchBlvmhBluxuryBlutonBlurchesBlungeBlungBlundgrenBlukeBluisBludovicoB	lucrativeBluckBlucidBlucianoBlucasBlualuaBltdBloynesBloyalBlowkeyBlowinflationBloweredBlovelyBlousyBlouiseBloudBlostprophetsBlorraineBlootingBloopsBlongwellBlongstandingB	longrangeBlonghornB	longevityBlongdenB
longcourseBlongawaitedBlonelyB	londonersBlondonderryBlondonbasedBlombardiB
logisticalBloginBloggingBloggedBloftusBlockingBlockedB	locationsBlocatedBlobbyingB
loanforoilBloBlloydBlivelyB
liveactionBliuBlittleknownBliteracyB	listeningB	liquidityB	liquefiedBlipstickBlinvoyBlinkingBlineoutBlinecallBlimitationsB
limitationBlimbsB
likhotsevaBlikenedB
likemindedB
likelihoodBlightsBlightmovingBlightlyBligamentBligaBliftsB	lifestyleBlifelikeBlifedrainingB	lifebloodB
lieutenantBliedBlickBlicensedBlicencesBlibyaB	librariesB
liberatingBliangBlianBliableBliabilitiesBlgB	levellingBlesserBlessensBleonciaBleonardoBlendBleicestershireBlehmannBlegsB
legisationB	legendaryBlegalisationBleewayBlecturerBleblancBleaseBleakingBleakedBleaguesBleafletsBleadupBleadenfootedBleBlciBlayoutBlayerBlawrenceBlawfulBlavagnaBlausanneBlaurentBlaurenBlatterBlatenovemberBlatchedB
lastminuteBlastingBlastedB	lastditchBlashedBlashBlasersBlaptopBlankaBlanguishingB
languishedB	languagesBlanesB
landscapesB
landlockedB
lancashireBlanBlampardBlaminateBlamBlakeBlairgBlahoreBladBlabsBlabouredBlaborBlabelsBlabelBlabBkuwaitBkunzruBkromeB	kozlowskiBkowalskiBkostasBkoogleBkoloBknocksBknitsBknifeBkneesB	kleinwortBkittsBkissBkisrtyBkinrossB	kingsholmBkingsB
kingfisherBkingdomBkimberlyB
kilometresB	kilometreBkillingBkillersBkillerBkillamangiroBkillBkilbrideBkikiBkieferBkidneyBkicksBkickoffBkickableB
khristenkoBkeyobsBkeskarBkenyanBkenyaBkentBkelvinBkeatonBkathyB	katherineBkaterinaBkartBkarstadtquelleBkarolBkarenBkansasBkalushaBk2BjuvenileBjustineBjustinB
justifyingBjurgenBjuniorBjungleBjumpyBjumpsBjumboBjulioBjulieBjuliaBjulesBjuiveBjuicesBjudiB
judgementsBjudgedBjudasBjuanBjoystickBjoypadBjoyousBjoshBjorgBjordanBjollyBjoleneBjoleBjokedBjointlyBjoieBjoiceBjoeyBjiulinBjieBjesperBjerwoodBjerryBjenkinsBjemmaBjelenaBjeffBjeanBjazzBjayBjaxxBjawsBjawdroppingBjankovicBjaneBjalBjakartabasedBjacobaBjacobBjacktheBjackieBjacketBiyadBivoBiunitBitpcBitemsBitemBissuingBisraeliBislandsBisaeBirvineBirreversibleB
irrelevantBirregularityB	irregularBironsB
ironicallyBironBirkBirisBirfanBipswichBipodtypeBipoBiphotoBinwhichB	involvingBinvoiceB
invigorateB	investingBinvestigativeBinvestigatingBinvadersB	inundatedBinuitB	intrusionBintrospectiveBintoppaBintimidatedB
interviewsB
intervenedB	interveneBintervalB
intertrustBinterruptingB	interruptBinterpretingBinterpreterB	interpretBinterpolBinteroperableB
internmentB	interneesBinternationallyBinterimBinterfaxBinterestsensitiveBinterconnectB
interceptsBinteractiveB
interactedBinteractBinterBintentionallyB	intensiveBintensifiedB	intenselyBinsurerBinsultsBinstrumentsBinstrumentalB
instructorBinstructB	installedB
inspectorsB	inspectedB	insolventB
insistenceBinsiderBinsensitiveBinquisitivenessBinputBinnovationsBinnerBinmatesBinktomiB	injusticeBinjurydepletedB
injunctionBinjectBinheritanceBinghamBingameBinfuriatingBinfringementB	infringedB	informingBinformalB
infogramesBinfluentialB
influencesB
influencedBinflateBinfirmB	infectionBinfantsBinexperiencedBinequalitiesBindulgeBindirectBindigentB
indigenousBindieBindictmentsB
indictmentB
indicativeB
indicatingBindexlinkedBindepthBindependentlyBincompatibleBincompatibilitiesBincitingBinchB
incapacityB	incapableBinbuiltBinboxesB
inadequateBinaccessibleB	inabilityBimprisoningBimpressivelyB
impressingB	impressesBimpressB
impoverishBimploredBimplicationsBimperialB
imperativeBimpededBimpBimovieB	immersiveBimmersesBimmaculatelyB
imbalancesBimaginationBimacB
illustrateB
illuminatiB	illsuitedBillserveBillegitimateBillBilifeBignoringBignoresBifpiBifootBieuanBie7BidvdBidowuB	idolstyleBidolsBidioticBidentificationB	identicalBidcBidBiconBicebergBiceBhypnoticBhypedBhyltonpottsBhvbB
hutchinsonBhustleBhusbandsBhurtsBhurlingB	hungarianBhungBhundredBhumbledBhumanfactorsBhulkBhugillBhughesBhueyBhubBhoyteBhoversBhoveringBhoveredBhostingBhostageBhorrorBhorizonsBhorizonBhordesBhopwoodBhopmanBhookedBhookB	honouringBhonourBhonestyBhonestlyBhometownB
homeownersBhomelessnessBhomelessB	homegrownBholdingsBhogwartsBhoggedBhoffmanBhobbyBhivaidsBhittingB
historicalBhislopBhiringBhiredBhippyBhintsBhintingBhinckleyBhiltonBhillsboroughB
hillingdonB	hilariousBhikesB
highvolumeB
highstreetBhighsBhighrankingBhighlyqualifiedBhighlyanticipatedB	highlevelBhighestgrossingBhigherqualityB	highclassBhighcapacityBhigginsBhifiBhidesBhickoxBhewlettB
hesitatingBherveBhertrichBheroBherdedBherbertBheraldsBhenleyonthamesBheninhardenneBhelsinkiBhelperBhellboyB
helicopterBhelgueraBhelenB	heightensBheightB
heerenveenBheelsBheelBhecubaBhecticBheavenBheatsBheatherBheathBheatedBheartsBhealthyBhealthrelatedB	healthierBheadsetBheadonBheadingBhclBhboBhazelBhayleyBhayaoBhauntingBhauledBhatfiendBhateBhassleBhassellBhassallBharshBharrietBharperBharmlessBharmfulBharmanBhariBhareBhardworkingB
hardfoughtBhardestBhardcoreBhappilyBhaplessBhanoverBhanningfieldBhanifBhangoverBhangedB
hangarlikeBhangBhandyBhandoutB	handheldsBhandanimatedB	hammertonBhammersBhamletBhalvedBhalveBhallsBhallowedB
halfvolleyBhalfhourBhalfheartedBhalevyBhakBhaitiBhaishengBhairdressingBhaircutBhairBhailBhadnBhacksBhacanBhabitsBhabeusB
gyllenhaalBgutterBgutsBgunshotBgunBgulfBguitartotingBguideBguessedBguessB
gudjohnsenBguardsBgruntBgrumbletextcoukB	gruellingBgrubbyB
grovellingBgroveBgroundstrokesBgroundsBgroundrulesBgroundbreakingBgrosjeanBgroinB	groceriesBgrittyBgritBgripsBgripesBgrimBgriffinBgriffenBgridlockB	greenwichBgreenerBgreeneBgreeceBgrecoBgreatlyBgrazBgrayB
gravestoneB
gratuitousBgrassB	grapplingBgraphicBgrandchildrenBgrammarBgraftB	graduallyBgradualBgradesBgradeBgraceBgpsBgprsBgpB	governorsBgovernedBgoughBgosportBgoodsellingBgoodeBgongBgondryBgolfBgoldmanB	goforwardBgodB	goalpostsB
goalkeeperBgoaBgmtvBgmbBgmailBglutBgluedBglowingB
gloriouslyBgloballyB
glitteringBglitchBglimpsesBglimmerBglenmorangieBgleesonBglasgowB	glaringlyBglaringBglanceBglamourBglamBgizmondoBgiveawayBgiuseppeBgiscardBgirvanBginoBgilmoreBgilesBgigglesB	gigabytesBgiftBgidleyB	giampaoloBgiacomoBgertzBgeordanB	genuinelyBgenresBgeneticallymodifiedBgenerosoB
generatingB	generatesBgeneralisedB
genemationBgeijsenBgeeksBgeeBgearsBgearedBgearBgdcBgaymardBgaugeBgathersB	gatherersBgatewayB
gastropodsBgaspBgarrosBgarlandBgardenB
garagebandBgarageBgapsBgamesindustrybizBgamebreakersBgambonBgalwaybasedBgalliumBgalanBgaizkaBgaggingBgabrieleBgabrielBg7B
futurologyB
futuristicBfurthermoreBfuroreB	furnitureBfurnishBfuquaBfunnyB
funnellingBfunkyBfundraisingB
fundraiserBfundingBfundedBfundamentallyB
functionalBfunctionBfumingB
fulllengthBfullestBfuerteventuraBftseBftBfrustrationB
frustratedBfruitsBfroogleBfrontsBfrontrowBfrontmanB	frontlineBfrontierBfrontedB	fromentalBfrodoBfrightenB
friendshipBfriedBfridgesBfrettedBfrequenciesBfrenziedBfreerangingBfreerB	freelanceBfreefallBfreedomfightingBfreddyBfredBfraughtB
fraudulentB	fraudsterB
fraternityBfraserBfrankjuergenB	frankfurtBfrancisB
franchisesB	frameworkB
fragmentedBfragmentationBfragileBfractureBfoxxBfouryearoldsBfouryearBfourwheeledBfourtierBfourthroundBfourthquarterB	fourmonthBfoundationsBforumBfortunesBfortuneBfortunatelyBfortressB	fortnightBforsythBforsterBformerlyB	formalityB	forgivingBforgingB
forevermanBforeverBforemanBforeignbasedBforecastingB
forecasterB
forebodingBfordB	footstepsBfootballplayingBfootballersB
footballerB	fondationBfomcBfollowupBfoldingBfoldBfoesB	focussingBfocusesBflyBflutterBflushBfluorescentBfluentBflowsBflowerBfloutingBfloutB
flourishesBflopBfloodedBfloodBfloatingBflightsBflightBflickBflicBfletcherBflatrateB	flatpanelBflatBflakBflaggedBfixingB	fixedlineBfixateBfiveyeardealBfiveweekB
fivestrongBfiveandahalfBfittingsBfittedBfitnessBfistB	firsthalfB	fireworksBfireworkBfirewallBfiresB
firefightsBfiproB
fionnuallaBfinsihedBfinnishBfiniteBfinianBfingerprintingBfingerBfindlayB	financierB	finalisedBfinaliseBfinaleBfiltersBfillerBfileswappingBfilderstadtBfijiB
figureheadBfighterB	fightbackBfifthplacedBfiestaBfieldsBfiddledBfiddleB	fictionalBfiascoB	feyenoordBferryBferreroBferreiraBferozBferociouslyB	ferociousBfergusonBfelonyBfelipeB	felicianoBfelgateBfeesBfeedbackBfeedBfeaturedBfeatherBfeatBfeasibleBfayeBfavoursBfavouredB
favourablyB
favourableBfavaBfaustoBfaulknerBfatigueBfathiBfasttrackingBfastgrowingBfastfoodBfastestBfashionableBfascinatingBfarrellBfarquharBfareBfangBfanaticsBfamouslyBfamiliarB
falsifyingB
fallingoutBfakingBfakeBfairnessBfairerBfailingsBfailBfactoryBfactorsBfabriceBfabregasBfabienBf16BfBeyeingBeyebrowsBeyB	extratimeBextraordinarilyB
extractionBextractBexternalBexpungedB
expressionBexposureB
expositionB	exportersB	exploringBexploredBexploreB	explodingBexplodedBexplicitB
expletivesBexpiredBexpireB	expertiseB
experimentBexpenseB	expansiveBexpandedB
expandableBexoticBexloverBexitsBexitnBexistsBexhomeBexhilaratingB
exhibitorsBexhibitionsB
exhibitingBexecutedB	exclusiveB	excludingB	exchequerBexchatB
exchangingB	excessiveBexcessBexcerptBexceedsBexceededBexbbcBexaneB	examiningBevolvingBevolvesBevolvedBevolveBevokesBevictBeveshamB	evergreenBeveBevanB	evalutionB	evacuatedBevaBeuwideBeustonBeuroscepticismB
europhilesBetienneBethnicitiesBetcBestimateBestateBestaingBestablishingB
essentialsBessenceBescapedBeruptedBerrantBerodedBericBerelBerasBequipBequatedBequalledB	equaliserB	eponymousB
epitomisedBepidemicBenvironmentallyB
envelopingBenvelopeBenvelopBentitlesBenthusiastsB
enthusiasmBentertainingBentertainersB
enterpriseBenjoyingBenjoyedB	enigmaticBengagedB	enforcingBenforcerBenforcementBenfanceBenergysappingBenemyBendtoendB	endowmentBendingsBendgameBendangeringBendacottBencasedBencapsulateBenactedBenablingB	emulatorsBemptyhandedBempowermentB	employingBemmersonBemmanuelBemiratesB	eminentlyBemiliaBemergesBembraceBembassyBelliotB	elizabethBelitistBeliotBeliminatingBelicitsBelevenB	elephantsBelephantBeleneBeleganceBelectricBelapsedB	elaborateBeinarBeightplayerBeightminuteB
eighthseedB
eighthorseBegwinBefficientlyBefficienciesBeffectivenessBeerieBeeghenBeducationalBeducatedBeduBednaBeditionsBeditionBeditedBedgyB	edgescapeBedgedBedB	economiesBechoedBechoB
eccentricsB	eccentricBebbsBeatingBeaterB
easytomakeBeasiestBeasesBeasedBearthBearsBearringBearningBearnedBearlyseasonBearliestBeaglesBe3BeBdynamiteBdynamismBdynamicBdwindleBdutiesB	dutchbornBdustinBdurbanBdurablesBdunkinBdummyBdullBdugBduffyBduesBduelBdudleyBdudasBduckingBdubbingBduaneBdslBdrunksBdrunkenBdrunkardBdruidBdroveBdrossBdrkwBdrivesBdrillBdreyfussBdressBdresdnerBdreddB
dreamlinerBdrawsBdrawnupBdrainageBdragonsBdraftsBdownthelineBdownshiftingBdownloadersBdownlinkB
downlightsB
downgradesBdourBdoublingBdoubleheaderBdoublebreakB
doubleblowBdotlifeBdoseBdorriesBdopingBdonutsB	donnellonBdonnedBdonkeyB	doncasterBdonateB
dominicansBdominicB	dominatesBdominateBdominantB	dominanceBdomainsBdomainBdolingBdoldrumsBdohaBdodgingBdodgeBdocumentarymakerBdockingBdocBdoblasBdoakBdjibrilBdiyBdivorcedB	divisionsBdivingBdivideBdiversifiesBdiverseB
disturbingBdistributorBdistributedBdistractionBdistinguishedBdistinguishB
distinctlyBdistinctB
distancingB	distancesB
dissuadingB	dissenterBdisputedB
disposableBdisplaysB
displayingB	displayedB	displacedB	dispensedBdispensationBdispassionateB	disordersBdisobeysBdismountingB
dismissiveB
dismissingBdismissBdislikedBdislikeB	dishonestB	disgracedB
discussionB
discussingBdiscriminatesBdiscreetB
discoveredB
discountedB
discontentB
disconnectB
discomfortB
disclosureB
disclosingB	disclosedBdiscB	disastersBdisappearingBdisagreementsB	directorsBdirectorialB
directivesBdippingBdinnersBdinkierBdinkedBdineshBdimmableB	dimensionBdilemmaBdigsBdignityB	digitisedBdigitBdigestB	differingBdifferentiateBdifferencesBdifferBdiehardBdictateBdickBdibabaBdiaryBdiariesBdianeBdialupBdialogB	diagnosedBdiageoBdevotedBdevoteBdevisedBdeviseBdevilsBdeuceB	detonated
╩ё
Const_9Const*
_output_shapes	
:ОN*
dtype0	*Мё
valueБёB¤Ё	ОN"ЁЁ                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              А       Б       В       Г       Д       Е       Ж       З       И       Й       К       Л       М       Н       О       П       Р       С       Т       У       Ф       Х       Ц       Ч       Ш       Щ       Ъ       Ы       Ь       Э       Ю       Я       а       б       в       г       д       е       ж       з       и       й       к       л       м       н       о       п       ░       ▒       ▓       │       ┤       ╡       ╢       ╖       ╕       ╣       ║       ╗       ╝       ╜       ╛       ┐       └       ┴       ┬       ├       ─       ┼       ╞       ╟       ╚       ╔       ╩       ╦       ╠       ═       ╬       ╧       ╨       ╤       ╥       ╙       ╘       ╒       ╓       ╫       ╪       ┘       ┌       █       ▄       ▌       ▐       ▀       р       с       т       у       ф       х       ц       ч       ш       щ       ъ       ы       ь       э       ю       я       Ё       ё       Є       є       Ї       ї       Ў       ў       °       ∙       ·       √       №       ¤       ■                                                                      	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■              	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      А	      Б	      В	      Г	      Д	      Е	      Ж	      З	      И	      Й	      К	      Л	      М	      Н	      О	      П	      Р	      С	      Т	      У	      Ф	      Х	      Ц	      Ч	      Ш	      Щ	      Ъ	      Ы	      Ь	      Э	      Ю	      Я	      а	      б	      в	      г	      д	      е	      ж	      з	      и	      й	      к	      л	      м	      н	      о	      п	      ░	      ▒	      ▓	      │	      ┤	      ╡	      ╢	      ╖	      ╕	      ╣	      ║	      ╗	      ╝	      ╜	      ╛	      ┐	      └	      ┴	      ┬	      ├	      ─	      ┼	      ╞	      ╟	      ╚	      ╔	      ╩	      ╦	      ╠	      ═	      ╬	      ╧	      ╨	      ╤	      ╥	      ╙	      ╘	      ╒	      ╓	      ╫	      ╪	      ┘	      ┌	      █	      ▄	      ▌	      ▐	      ▀	      р	      с	      т	      у	      ф	      х	      ц	      ч	      ш	      щ	      ъ	      ы	      ь	      э	      ю	      я	      Ё	      ё	      Є	      є	      Ї	      ї	      Ў	      ў	      °	      ∙	      ·	      √	      №	      ¤	      ■	       	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      А
      Б
      В
      Г
      Д
      Е
      Ж
      З
      И
      Й
      К
      Л
      М
      Н
      О
      П
      Р
      С
      Т
      У
      Ф
      Х
      Ц
      Ч
      Ш
      Щ
      Ъ
      Ы
      Ь
      Э
      Ю
      Я
      а
      б
      в
      г
      д
      е
      ж
      з
      и
      й
      к
      л
      м
      н
      о
      п
      ░
      ▒
      ▓
      │
      ┤
      ╡
      ╢
      ╖
      ╕
      ╣
      ║
      ╗
      ╝
      ╜
      ╛
      ┐
      └
      ┴
      ┬
      ├
      ─
      ┼
      ╞
      ╟
      ╚
      ╔
      ╩
      ╦
      ╠
      ═
      ╬
      ╧
      ╨
      ╤
      ╥
      ╙
      ╘
      ╒
      ╓
      ╫
      ╪
      ┘
      ┌
      █
      ▄
      ▌
      ▐
      ▀
      р
      с
      т
      у
      ф
      х
      ц
      ч
      ш
      щ
      ъ
      ы
      ь
      э
      ю
      я
      Ё
      ё
      Є
      є
      Ї
      ї
      Ў
      ў
      °
      ∙
      ·
      √
      №
      ¤
      ■
       
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                    	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            А      Б      В      Г      Д      Е      Ж      З      И      Й      К      Л      М      Н      О      П      Р      С      Т      У      Ф      Х      Ц      Ч      Ш      Щ      Ъ      Ы      Ь      Э      Ю      Я      а      б      в      г      д      е      ж      з      и      й      к      л      м      н      о      п      ░      ▒      ▓      │      ┤      ╡      ╢      ╖      ╕      ╣      ║      ╗      ╝      ╜      ╛      ┐      └      ┴      ┬      ├      ─      ┼      ╞      ╟      ╚      ╔      ╩      ╦      ╠      ═      ╬      ╧      ╨      ╤      ╥      ╙      ╘      ╒      ╓      ╫      ╪      ┘      ┌      █      ▄      ▌      ▐      ▀      р      с      т      у      ф      х      ц      ч      ш      щ      ъ      ы      ь      э      ю      я      Ё      ё      Є      є      Ї      ї      Ў      ў      °      ∙      ·      √      №      ¤      ■                                                                             	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              А       Б       В       Г       Д       Е       Ж       З       И       Й       К       Л       М       Н       О       П       Р       С       Т       У       Ф       Х       Ц       Ч       Ш       Щ       Ъ       Ы       Ь       Э       Ю       Я       а       б       в       г       д       е       ж       з       и       й       к       л       м       н       о       п       ░       ▒       ▓       │       ┤       ╡       ╢       ╖       ╕       ╣       ║       ╗       ╝       ╜       ╛       ┐       └       ┴       ┬       ├       ─       ┼       ╞       ╟       ╚       ╔       ╩       ╦       ╠       ═       ╬       ╧       ╨       ╤       ╥       ╙       ╘       ╒       ╓       ╫       ╪       ┘       ┌       █       ▄       ▌       ▐       ▀       р       с       т       у       ф       х       ц       ч       ш       щ       ъ       ы       ь       э       ю       я       Ё       ё       Є       є       Ї       ї       Ў       ў       °       ∙       ·       √       №       ¤       ■                !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      А!      Б!      В!      Г!      Д!      Е!      Ж!      З!      И!      Й!      К!      Л!      М!      Н!      О!      П!      Р!      С!      Т!      У!      Ф!      Х!      Ц!      Ч!      Ш!      Щ!      Ъ!      Ы!      Ь!      Э!      Ю!      Я!      а!      б!      в!      г!      д!      е!      ж!      з!      и!      й!      к!      л!      м!      н!      о!      п!      ░!      ▒!      ▓!      │!      ┤!      ╡!      ╢!      ╖!      ╕!      ╣!      ║!      ╗!      ╝!      ╜!      ╛!      ┐!      └!      ┴!      ┬!      ├!      ─!      ┼!      ╞!      ╟!      ╚!      ╔!      ╩!      ╦!      ╠!      ═!      ╬!      ╧!      ╨!      ╤!      ╥!      ╙!      ╘!      ╒!      ╓!      ╫!      ╪!      ┘!      ┌!      █!      ▄!      ▌!      ▐!      ▀!      р!      с!      т!      у!      ф!      х!      ц!      ч!      ш!      щ!      ъ!      ы!      ь!      э!      ю!      я!      Ё!      ё!      Є!      є!      Ї!      ї!      Ў!      ў!      °!      ∙!      ·!      √!      №!      ¤!      ■!       !       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      А"      Б"      В"      Г"      Д"      Е"      Ж"      З"      И"      Й"      К"      Л"      М"      Н"      О"      П"      Р"      С"      Т"      У"      Ф"      Х"      Ц"      Ч"      Ш"      Щ"      Ъ"      Ы"      Ь"      Э"      Ю"      Я"      а"      б"      в"      г"      д"      е"      ж"      з"      и"      й"      к"      л"      м"      н"      о"      п"      ░"      ▒"      ▓"      │"      ┤"      ╡"      ╢"      ╖"      ╕"      ╣"      ║"      ╗"      ╝"      ╜"      ╛"      ┐"      └"      ┴"      ┬"      ├"      ─"      ┼"      ╞"      ╟"      ╚"      ╔"      ╩"      ╦"      ╠"      ═"      ╬"      ╧"      ╨"      ╤"      ╥"      ╙"      ╘"      ╒"      ╓"      ╫"      ╪"      ┘"      ┌"      █"      ▄"      ▌"      ▐"      ▀"      р"      с"      т"      у"      ф"      х"      ц"      ч"      ш"      щ"      ъ"      ы"      ь"      э"      ю"      я"      Ё"      ё"      Є"      є"      Ї"      ї"      Ў"      ў"      °"      ∙"      ·"      √"      №"      ¤"      ■"       "       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      А#      Б#      В#      Г#      Д#      Е#      Ж#      З#      И#      Й#      К#      Л#      М#      Н#      О#      П#      Р#      С#      Т#      У#      Ф#      Х#      Ц#      Ч#      Ш#      Щ#      Ъ#      Ы#      Ь#      Э#      Ю#      Я#      а#      б#      в#      г#      д#      е#      ж#      з#      и#      й#      к#      л#      м#      н#      о#      п#      ░#      ▒#      ▓#      │#      ┤#      ╡#      ╢#      ╖#      ╕#      ╣#      ║#      ╗#      ╝#      ╜#      ╛#      ┐#      └#      ┴#      ┬#      ├#      ─#      ┼#      ╞#      ╟#      ╚#      ╔#      ╩#      ╦#      ╠#      ═#      ╬#      ╧#      ╨#      ╤#      ╥#      ╙#      ╘#      ╒#      ╓#      ╫#      ╪#      ┘#      ┌#      █#      ▄#      ▌#      ▐#      ▀#      р#      с#      т#      у#      ф#      х#      ц#      ч#      ш#      щ#      ъ#      ы#      ь#      э#      ю#      я#      Ё#      ё#      Є#      є#      Ї#      ї#      Ў#      ў#      °#      ∙#      ·#      √#      №#      ¤#      ■#       #       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      А$      Б$      В$      Г$      Д$      Е$      Ж$      З$      И$      Й$      К$      Л$      М$      Н$      О$      П$      Р$      С$      Т$      У$      Ф$      Х$      Ц$      Ч$      Ш$      Щ$      Ъ$      Ы$      Ь$      Э$      Ю$      Я$      а$      б$      в$      г$      д$      е$      ж$      з$      и$      й$      к$      л$      м$      н$      о$      п$      ░$      ▒$      ▓$      │$      ┤$      ╡$      ╢$      ╖$      ╕$      ╣$      ║$      ╗$      ╝$      ╜$      ╛$      ┐$      └$      ┴$      ┬$      ├$      ─$      ┼$      ╞$      ╟$      ╚$      ╔$      ╩$      ╦$      ╠$      ═$      ╬$      ╧$      ╨$      ╤$      ╥$      ╙$      ╘$      ╒$      ╓$      ╫$      ╪$      ┘$      ┌$      █$      ▄$      ▌$      ▐$      ▀$      р$      с$      т$      у$      ф$      х$      ц$      ч$      ш$      щ$      ъ$      ы$      ь$      э$      ю$      я$      Ё$      ё$      Є$      є$      Ї$      ї$      Ў$      ў$      °$      ∙$      ·$      √$      №$      ¤$      ■$       $       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      А%      Б%      В%      Г%      Д%      Е%      Ж%      З%      И%      Й%      К%      Л%      М%      Н%      О%      П%      Р%      С%      Т%      У%      Ф%      Х%      Ц%      Ч%      Ш%      Щ%      Ъ%      Ы%      Ь%      Э%      Ю%      Я%      а%      б%      в%      г%      д%      е%      ж%      з%      и%      й%      к%      л%      м%      н%      о%      п%      ░%      ▒%      ▓%      │%      ┤%      ╡%      ╢%      ╖%      ╕%      ╣%      ║%      ╗%      ╝%      ╜%      ╛%      ┐%      └%      ┴%      ┬%      ├%      ─%      ┼%      ╞%      ╟%      ╚%      ╔%      ╩%      ╦%      ╠%      ═%      ╬%      ╧%      ╨%      ╤%      ╥%      ╙%      ╘%      ╒%      ╓%      ╫%      ╪%      ┘%      ┌%      █%      ▄%      ▌%      ▐%      ▀%      р%      с%      т%      у%      ф%      х%      ц%      ч%      ш%      щ%      ъ%      ы%      ь%      э%      ю%      я%      Ё%      ё%      Є%      є%      Ї%      ї%      Ў%      ў%      °%      ∙%      ·%      √%      №%      ¤%      ■%       %       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      А&      Б&      В&      Г&      Д&      Е&      Ж&      З&      И&      Й&      К&      Л&      М&      Н&      О&      П&      Р&      С&      Т&      У&      Ф&      Х&      Ц&      Ч&      Ш&      Щ&      Ъ&      Ы&      Ь&      Э&      Ю&      Я&      а&      б&      в&      г&      д&      е&      ж&      з&      и&      й&      к&      л&      м&      н&      о&      п&      ░&      ▒&      ▓&      │&      ┤&      ╡&      ╢&      ╖&      ╕&      ╣&      ║&      ╗&      ╝&      ╜&      ╛&      ┐&      └&      ┴&      ┬&      ├&      ─&      ┼&      ╞&      ╟&      ╚&      ╔&      ╩&      ╦&      ╠&      ═&      ╬&      ╧&      ╨&      ╤&      ╥&      ╙&      ╘&      ╒&      ╓&      ╫&      ╪&      ┘&      ┌&      █&      ▄&      ▌&      ▐&      ▀&      р&      с&      т&      у&      ф&      х&      ц&      ч&      ш&      щ&      ъ&      ы&      ь&      э&      ю&      я&      Ё&      ё&      Є&      є&      Ї&      ї&      Ў&      ў&      °&      ∙&      ·&      √&      №&      ¤&      ■&       &       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
Ю
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_8Const_9*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *#
fR
__inference_<lambda>_21223
щ
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *#
fR
__inference_<lambda>_21228
e
ReadVariableOpReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
и
StatefulPartitionedCall_2StatefulPartitionedCallReadVariableOpStatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *#
fR
__inference_<lambda>_21246
|
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^Variable/Assign^Variable_1/Assign
╟
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
ўD
Const_10Const"/device:CPU:0*
_output_shapes
: *
dtype0*пD
valueеDBвD BЫD
╥
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

	optimizer
		tft_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

	keras_api* 
;
_lookup_layer
	keras_api
_adapt_function*
а

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
О
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
ж

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
ж

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
ж

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
┤
$< _saved_model_loader_tracked_dict
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
╨
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratemЦ$mЧ%mШ,mЩ-mЪ4mЫ5mЬvЭ$vЮ%vЯ,vа-vб4vв5vг*
5
1
$2
%3
,4
-5
46
57*
5
0
$1
%2
,3
-4
45
56*
* 
░
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Mserving_default* 
* 
7
Nlookup_table
Otoken_counts
P	keras_api*
* 
* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
У
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
С
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
У
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 
У
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 
У
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
t
j	_imported
k_structured_inputs
l_structured_outputs
m_output_to_inputs_map
n_wrapped_function* 
* 
* 
* 
С
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

t0
u1*
* 
* 
* 
R
v_initializer
w_create_resource
x_initialize
y_destroy_resource* 
О
z_create_resource
{_initialize
|_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
и
}created_variables
~	resources
trackable_objects
Аinitializers
Бassets
В
signatures
$Г_self_saveable_object_factories
ntransform_fn* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

Дtotal

Еcount
Ж	variables
З	keras_api*
M

Иtotal

Йcount
К
_fn_kwargs
Л	variables
М	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 


Н0* 
* 


О0* 


П0* 

Рserving_default* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Д0
Е1*

Ж	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

И0
Й1*

Л	variables*
V
О_initializer
С_create_resource
Т_initialize
У_destroy_resource* 
8
Ф	_filename
$Х_self_saveable_object_factories* 
* 
* 
* 
* 
* 
* 
* 
МЕ
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
МЕ
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
s
serving_default_examplesPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
Э
StatefulPartitionedCall_3StatefulPartitionedCallserving_default_examplesConst_4Const_5StatefulPartitionedCallConst_6Const_7
hash_tableConstConst_1Const_2embedding/embeddingsdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *)
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_20129
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
░
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst_10*-
Tin&
$2"		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_21392
▌
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/embedding/embeddings/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_21495ух
еn
е
B__inference_model_1_layer_call_and_return_conditional_losses_20728
text_xfU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_20708:	РN 
dense_3_20712:	└
dense_3_20714:	└!
dense_4_20717:
└А
dense_4_20719:	А 
dense_5_20722:	А
dense_5_20724:
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвDtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         {
tf.reshape_1/ReshapeReshapetext_xf#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:         ╨
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ┌
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Г
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Е
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╤
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╚
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ш
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:й
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: █
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: е
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ф
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ¤
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: л
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: б
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :┘
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ═
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ╤
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: д
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ╩
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : с
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         к
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         є
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╛
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         Л
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         Ч
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R А
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       Ж
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSЫ
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_20708*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20383 
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_20221Ч
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_3_20712dense_3_20714*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_20399М
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_20717dense_4_20719*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_20416Л
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_20722dense_5_20724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_20433w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ч
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2М
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:         
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ё
╢
__inference_pruned_19889

inputs
inputs_1:
6compute_and_apply_vocabulary_vocabulary_identity_input	<
8compute_and_apply_vocabulary_vocabulary_identity_1_input	W
Scompute_and_apply_vocabulary_apply_vocab_none_lookup_lookuptablefindv2_table_handleX
Tcompute_and_apply_vocabulary_apply_vocab_none_lookup_lookuptablefindv2_default_value	2
.compute_and_apply_vocabulary_apply_vocab_sub_x	
identity	

identity_1ИQ
inputs_copyIdentityinputs*
T0*'
_output_shapes
:         ═
Fcompute_and_apply_vocabulary/apply_vocab/None_Lookup/LookupTableFindV2LookupTableFindV2Scompute_and_apply_vocabulary_apply_vocab_none_lookup_lookuptablefindv2_table_handleinputs_copy:output:0Tcompute_and_apply_vocabulary_apply_vocab_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*
_output_shapes
:O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  А?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ╘
one_hotOneHotOcompute_and_apply_vocabulary/apply_vocab/None_Lookup/LookupTableFindV2:values:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       n
ReshapeReshapeone_hot:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:         _
CastCastReshape:output:0*

DstT0	*

SrcT0*'
_output_shapes
:         П
NoOpNoOpG^compute_and_apply_vocabulary/apply_vocab/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityCast:y:0^NoOp*
T0	*'
_output_shapes
:         U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:         [
StringLowerStringLowerinputs_1_copy:output:0*'
_output_shapes
:         e

Identity_1IdentityStringLower:output:0^NoOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         :         : : : : : :- )
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Н
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_21049

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
▄
U
(__inference_restored_function_body_21285
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *#
fR
__inference__creator_19938^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
т
q
(__inference_restored_function_body_21238
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__initializer_19867^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
Н	
╕
8__inference_transform_features_layer_layer_call_fn_20272
text
unknown	
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3	
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCalltextunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_20259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:         

_user_specified_nametext:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ю
┘
__inference_restore_fn_21215
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identityИв2MutableHashTable_table_restore/LookupTableImportV2Н
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
│y
д
B__inference_model_1_layer_call_and_return_conditional_losses_21022

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	3
 embedding_embedding_lookup_20993:	РN9
&dense_3_matmul_readvariableop_resource:	└6
'dense_3_biasadd_readvariableop_resource:	└:
&dense_4_matmul_readvariableop_resource:
└А6
'dense_4_biasadd_readvariableop_resource:	А9
&dense_5_matmul_readvariableop_resource:	А5
'dense_5_biasadd_readvariableop_resource:
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpвembedding/embedding_lookupвDtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:         ╨
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ┌
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Г
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Е
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╤
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╚
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ш
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:й
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: █
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: е
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ф
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ¤
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: л
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: б
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :┘
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ═
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ╤
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: д
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ╩
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : с
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         к
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         є
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╛
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         Л
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         Ч
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R А
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       Ж
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSО
embedding/embedding_lookupResourceGather embedding_embedding_lookup_20993?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/20993*+
_output_shapes
:         d*
dtype0┐
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/20993*+
_output_shapes
:         dХ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         ds
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :┼
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Е
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	└*
dtype0Ь
dense_3/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └Г
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype0П
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └a
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         └Ж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0О
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         АЕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         э
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2М
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ

с
'__inference_model_1_layer_call_fn_20465
text_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	РN
	unknown_4:	└
	unknown_5:	└
	unknown_6:
└А
	unknown_7:	А
	unknown_8:	А
	unknown_9:
identityИвStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCalltext_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_20440o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
МC
в
__inference_adapt_step_18554
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	ИвIteratorGetNextв(None_lookup_table_find/LookupTableFindV2в,None_lookup_table_insert/LookupTableInsertV2й
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:         *"
output_shapes
:         *
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:         к
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B б
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:         :         :p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskл
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         в
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ┬
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:Ц
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: в
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: Т
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : л
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ╫
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: Ш
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: У
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: О
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :а
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: У
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: Ф
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: Ш
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: С
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ■
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : и
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Ч
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         г
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:         :         :         *
out_idx0	б
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:Я
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
з
┬
__inference__initializer_19867!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityИв,text_file_init/InitializeTableFromTextFileV2є
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_index■        *
value_index         G
ConstConst*
_output_shapes
: *
dtype0*
value	B :u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
└
:
__inference__creator_19938
identityИв
hash_table▌

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*ш
shared_name╪╒hash_table_tf.Tensor(b'fajri27-pipeline\\fajri27-pipeline\\Transform\\transform_graph\\8\\.temp_path\\tftransform_tmp\\vocab_compute_and_apply_vocabulary_vocabulary', shape=(), dtype=string)_-2_-1_load_19857_19934*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
┴
Х
'__inference_dense_5_layer_call_fn_21098

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall╫
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_20433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
¤
V
:__inference_global_average_pooling1d_1_layer_call_fn_21043

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_20221i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Ь
╒
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_20259

inputs
unknown	
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3	
identityИвStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:L
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB B l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:         *
dtype0*
shape:         ф
StatefulPartitionedCallStatefulPartitionedCallPlaceholderWithDefault:output:0inputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2				*
Tout
2	*:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *!
fR
__inference_pruned_19889o
IdentityIdentity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╪
о
#__inference_signature_wrapper_20129
examples
unknown	
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3	
	unknown_4
	unknown_5	
	unknown_6
	unknown_7	
	unknown_8:	РN
	unknown_9:	└

unknown_10:	└

unknown_11:
└А

unknown_12:	А

unknown_13:	А

unknown_14:
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *)
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_serve_tf_examples_fn_20090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:         
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
─
б
D__inference_embedding_layer_call_and_return_conditional_losses_21038

inputs	)
embedding_lookup_21032:	РN
identityИвembedding_lookup╖
embedding_lookupResourceGatherembedding_lookup_21032inputs*
Tindices0	*)
_class
loc:@embedding_lookup/21032*+
_output_shapes
:         d*
dtype0б
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21032*+
_output_shapes
:         dБ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╖
c
__inference_<lambda>_21246
unknown
	unknown_0
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *1
f,R*
(__inference_restored_function_body_21238J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
╣
д
__inference_save_fn_21207
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	Ив?MutableHashTable_lookup_table_export_values/LookupTableExportV2М
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: И

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: К

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:И
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2В
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
е

Ў
B__inference_dense_4_layer_call_and_return_conditional_losses_21089

inputs2
matmul_readvariableop_resource:
└А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
б

ї
B__inference_dense_3_layer_call_and_return_conditional_losses_20399

inputs1
matmul_readvariableop_resource:	└.
biasadd_readvariableop_resource:	└
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:└*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         └b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         └w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ъ
,
__inference__destroyer_21173
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
З
№
!__inference__traced_restore_21495
file_prefix8
%assignvariableop_embedding_embeddings:	РN4
!assignvariableop_1_dense_3_kernel:	└.
assignvariableop_2_dense_3_bias:	└5
!assignvariableop_3_dense_4_kernel:
└А.
assignvariableop_4_dense_4_bias:	А4
!assignvariableop_5_dense_5_kernel:	А-
assignvariableop_6_dense_5_bias:&
assignvariableop_7_adam_iter:	 (
assignvariableop_8_adam_beta_1: (
assignvariableop_9_adam_beta_2: (
assignvariableop_10_adam_decay: 0
&assignvariableop_11_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: #
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: B
/assignvariableop_16_adam_embedding_embeddings_m:	РN<
)assignvariableop_17_adam_dense_3_kernel_m:	└6
'assignvariableop_18_adam_dense_3_bias_m:	└=
)assignvariableop_19_adam_dense_4_kernel_m:
└А6
'assignvariableop_20_adam_dense_4_bias_m:	А<
)assignvariableop_21_adam_dense_5_kernel_m:	А5
'assignvariableop_22_adam_dense_5_bias_m:B
/assignvariableop_23_adam_embedding_embeddings_v:	РN<
)assignvariableop_24_adam_dense_3_kernel_v:	└6
'assignvariableop_25_adam_dense_3_bias_v:	└=
)assignvariableop_26_adam_dense_4_kernel_v:
└А6
'assignvariableop_27_adam_dense_4_bias_v:	А<
)assignvariableop_28_adam_dense_5_kernel_v:	А5
'assignvariableop_29_adam_dense_5_bias_v:
identity_31ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в2MutableHashTable_table_restore/LookupTableImportV2Ў
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*Ь
valueТBП!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH▓
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╞
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ъ
_output_shapesЗ
Д:::::::::::::::::::::::::::::::::*/
dtypes%
#2!		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_3_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_3_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_4_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_5_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_5_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_11AssignVariableOp&assignvariableop_11_adam_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0М
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:12RestoreV2:tensors:13*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_16AssignVariableOp/assignvariableop_16_adam_embedding_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_3_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_3_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_4_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_4_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_5_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_3_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_3_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_4_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_4_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_5_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_5_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ш
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: Е
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
▌

├
#__inference_signature_wrapper_19902

inputs
inputs_1
unknown	
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3	
identity	

identity_1ИвStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2				*
Tout
2	*:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *!
fR
__inference_pruned_19889`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:         :         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
inputs_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ

с
'__inference_model_1_layer_call_fn_20656
text_xf
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	РN
	unknown_4:	└
	unknown_5:	└
	unknown_6:
└А
	unknown_7:	А
	unknown_8:	А
	unknown_9:
identityИвStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCalltext_xfunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_20604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
З
F
__inference__creator_21178
identity: ИвMutableHashTableА
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_13818*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
Ь
.
__inference__initializer_21183
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
─
б
D__inference_embedding_layer_call_and_return_conditional_losses_20383

inputs	)
embedding_lookup_20377:	РN
identityИвembedding_lookup╖
embedding_lookupResourceGatherembedding_lookup_20377inputs*
Tindices0	*)
_class
loc:@embedding_lookup/20377*+
_output_shapes
:         d*
dtype0б
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20377*+
_output_shapes
:         dБ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:         dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
├и
Й
&__inference_serve_tf_examples_fn_20090
examples"
transform_features_layer_20000	"
transform_features_layer_20002	"
transform_features_layer_20004"
transform_features_layer_20006	"
transform_features_layer_20008	]
Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle^
Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	4
0model_1_text_vectorization_string_lookup_equal_y7
3model_1_text_vectorization_string_lookup_selectv2_t	;
(model_1_embedding_embedding_lookup_20061:	РNA
.model_1_dense_3_matmul_readvariableop_resource:	└>
/model_1_dense_3_biasadd_readvariableop_resource:	└B
.model_1_dense_4_matmul_readvariableop_resource:
└А>
/model_1_dense_4_biasadd_readvariableop_resource:	АA
.model_1_dense_5_matmul_readvariableop_resource:	А=
/model_1_dense_5_biasadd_readvariableop_resource:
identityИв&model_1/dense_3/BiasAdd/ReadVariableOpв%model_1/dense_3/MatMul/ReadVariableOpв&model_1/dense_4/BiasAdd/ReadVariableOpв%model_1/dense_4/MatMul/ReadVariableOpв&model_1/dense_5/BiasAdd/ReadVariableOpв%model_1/dense_5/MatMul/ReadVariableOpв"model_1/embedding/embedding_lookupвLmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2в0transform_features_layer/StatefulPartitionedCallU
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB s
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*
valueBBtextj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB ├
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0*
Tdense
2*'
_output_shapes
:         *
dense_shapes
:*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 x
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :└
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB B ╖
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0*'
_output_shapes
:         ╞
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:         *
dtype0*
shape:         е
0transform_features_layer/StatefulPartitionedCallStatefulPartitionedCall8transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:0transform_features_layer_20000transform_features_layer_20002transform_features_layer_20004transform_features_layer_20006transform_features_layer_20008*
Tin
	2				*
Tout
2	*:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *!
fR
__inference_pruned_19889u
"model_1/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         ╜
model_1/tf.reshape_1/ReshapeReshape9transform_features_layer/StatefulPartitionedCall:output:1+model_1/tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         Б
&model_1/text_vectorization/StringLowerStringLower%model_1/tf.reshape_1/Reshape:output:0*#
_output_shapes
:         р
-model_1/text_vectorization/StaticRegexReplaceStaticRegexReplace/model_1/text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite m
,model_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B Є
4model_1/text_vectorization/StringSplit/StringSplitV2StringSplitV26model_1/text_vectorization/StaticRegexReplace:output:05model_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Л
:model_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Н
<model_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Н
<model_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
4model_1/text_vectorization/StringSplit/strided_sliceStridedSlice>model_1/text_vectorization/StringSplit/StringSplitV2:indices:0Cmodel_1/text_vectorization/StringSplit/strided_slice/stack:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЖ
<model_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: И
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:И
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
6model_1/text_vectorization/StringSplit/strided_slice_1StridedSlice<model_1/text_vectorization/StringSplit/StringSplitV2:shape:0Emodel_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskс
]model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast=model_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╪
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast?model_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: °
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:▒
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: є
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdpmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: н
kmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : №
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateromodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0tmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: Н
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastmmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: │
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ф
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: й
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :ё
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2nmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ф
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuljmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: х
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: щ
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: м
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ъ
jmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         ж
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ∙
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumqmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         ▓
hmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ж
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2qmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         У
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle=model_1/text_vectorization/StringSplit/StringSplitV2:values:0Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╓
.model_1/text_vectorization/string_lookup/EqualEqual=model_1/text_vectorization/StringSplit/StringSplitV2:values:00model_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         л
1model_1/text_vectorization/string_lookup/SelectV2SelectV22model_1/text_vectorization/string_lookup/Equal:z:03model_1_text_vectorization_string_lookup_selectv2_tUmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         з
1model_1/text_vectorization/string_lookup/IdentityIdentity:model_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         y
7model_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R И
/model_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       ╢
>model_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor8model_1/text_vectorization/RaggedToTensor/Const:output:0:model_1/text_vectorization/string_lookup/Identity:output:0@model_1/text_vectorization/RaggedToTensor/default_value:output:0?model_1/text_vectorization/StringSplit/strided_slice_1:output:0=model_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSо
"model_1/embedding/embedding_lookupResourceGather(model_1_embedding_embedding_lookup_20061Gmodel_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_1/embedding/embedding_lookup/20061*+
_output_shapes
:         d*
dtype0╫
+model_1/embedding/embedding_lookup/IdentityIdentity+model_1/embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model_1/embedding/embedding_lookup/20061*+
_output_shapes
:         dе
-model_1/embedding/embedding_lookup/Identity_1Identity4model_1/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         d{
9model_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▌
'model_1/global_average_pooling1d_1/MeanMean6model_1/embedding/embedding_lookup/Identity_1:output:0Bmodel_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Х
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	└*
dtype0┤
model_1/dense_3/MatMulMatMul0model_1/global_average_pooling1d_1/Mean:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └У
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype0з
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └q
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         └Ц
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0ж
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0з
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аq
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         АХ
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0е
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Т
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
model_1/dense_5/SoftmaxSoftmax model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         p
IdentityIdentity!model_1/dense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         р
NoOpNoOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp#^model_1/embedding/embedding_lookupM^model_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV21^transform_features_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:         : : : : : : : : : : : : : : : : 2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2H
"model_1/embedding/embedding_lookup"model_1/embedding/embedding_lookup2Ь
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22d
0transform_features_layer/StatefulPartitionedCall0transform_features_layer/StatefulPartitionedCall:M I
#
_output_shapes
:         
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
Т
╙
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_20318
text
unknown	
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3	
identityИвStatefulPartitionedCall9
ShapeShapetext*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask;
Shape_1Shapetext*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:L
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB B l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:         *
dtype0*
shape:         т
StatefulPartitionedCallStatefulPartitionedCallPlaceholderWithDefault:output:0textunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2				*
Tout
2	*:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *!
fR
__inference_pruned_19889o
IdentityIdentity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:         

_user_specified_nametext:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Н
q
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_20221

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
С
√
__inference__initializer_211688
4key_value_init15763_lookuptableimportv2_table_handle0
,key_value_init15763_lookuptableimportv2_keys2
.key_value_init15763_lookuptableimportv2_values	
identityИв'key_value_init15763/LookupTableImportV2 
'key_value_init15763/LookupTableImportV2LookupTableImportV24key_value_init15763_lookuptableimportv2_table_handle,key_value_init15763_lookuptableimportv2_keys.key_value_init15763_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init15763/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :ОN:ОN2R
'key_value_init15763/LookupTableImportV2'key_value_init15763/LookupTableImportV2:!

_output_shapes	
:ОN:!

_output_shapes	
:ОN
в

Ї
B__inference_dense_5_layer_call_and_return_conditional_losses_20433

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ч

р
'__inference_model_1_layer_call_fn_20833

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	РN
	unknown_4:	└
	unknown_5:	└
	unknown_6:
└А
	unknown_7:	А
	unknown_8:	А
	unknown_9:
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_20440o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
│y
д
B__inference_model_1_layer_call_and_return_conditional_losses_20941

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	3
 embedding_embedding_lookup_20912:	РN9
&dense_3_matmul_readvariableop_resource:	└6
'dense_3_biasadd_readvariableop_resource:	└:
&dense_4_matmul_readvariableop_resource:
└А6
'dense_4_biasadd_readvariableop_resource:	А9
&dense_5_matmul_readvariableop_resource:	А5
'dense_5_biasadd_readvariableop_resource:
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpвembedding/embedding_lookupвDtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:         ╨
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ┌
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Г
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Е
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╤
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╚
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ш
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:й
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: █
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: е
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ф
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ¤
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: л
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: б
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :┘
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ═
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ╤
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: д
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ╩
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : с
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         к
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         є
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╛
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         Л
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         Ч
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R А
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       Ж
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSО
embedding/embedding_lookupResourceGather embedding_embedding_lookup_20912?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*3
_class)
'%loc:@embedding/embedding_lookup/20912*+
_output_shapes
:         d*
dtype0┐
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/20912*+
_output_shapes
:         dХ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         ds
1global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :┼
global_average_pooling1d_1/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Е
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	└*
dtype0Ь
dense_3/MatMulMatMul(global_average_pooling1d_1/Mean:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └Г
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype0П
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └a
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         └Ж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0О
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         АЕ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         э
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2М
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
в	
┐
8__inference_transform_features_layer_layer_call_fn_21124
inputs_text
unknown	
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3	
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs_textunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *\
fWRU
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_20259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_nameinputs/text:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
вn
д
B__inference_model_1_layer_call_and_return_conditional_losses_20604

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_20584:	РN 
dense_3_20588:	└
dense_3_20590:	└!
dense_4_20593:
└А
dense_4_20595:	А 
dense_5_20598:	А
dense_5_20600:
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвDtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:         ╨
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ┌
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Г
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Е
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╤
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╚
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ш
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:й
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: █
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: е
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ф
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ¤
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: л
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: б
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :┘
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ═
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ╤
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: д
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ╩
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : с
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         к
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         є
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╛
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         Л
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         Ч
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R А
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       Ж
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSЫ
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_20584*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20383 
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_20221Ч
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_3_20588dense_3_20590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_20399М
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_20593dense_4_20595*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_20416Л
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_20598dense_5_20600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_20433w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ч
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2М
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч

р
'__inference_model_1_layer_call_fn_20860

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	РN
	unknown_4:	└
	unknown_5:	└
	unknown_6:
└А
	unknown_7:	А
	unknown_8:	А
	unknown_9:
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_20604o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┼
Ч
'__inference_dense_4_layer_call_fn_21078

inputs
unknown:
└А
	unknown_0:	А
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_20416p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
╧
:
__inference__creator_21160
identityИв
hash_tablem

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name15764*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
 Д
Ы	
 __inference__wrapped_model_20211
text_xf]
Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle^
Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	4
0model_1_text_vectorization_string_lookup_equal_y7
3model_1_text_vectorization_string_lookup_selectv2_t	;
(model_1_embedding_embedding_lookup_20182:	РNA
.model_1_dense_3_matmul_readvariableop_resource:	└>
/model_1_dense_3_biasadd_readvariableop_resource:	└B
.model_1_dense_4_matmul_readvariableop_resource:
└А>
/model_1_dense_4_biasadd_readvariableop_resource:	АA
.model_1_dense_5_matmul_readvariableop_resource:	А=
/model_1_dense_5_biasadd_readvariableop_resource:
identityИв&model_1/dense_3/BiasAdd/ReadVariableOpв%model_1/dense_3/MatMul/ReadVariableOpв&model_1/dense_4/BiasAdd/ReadVariableOpв%model_1/dense_4/MatMul/ReadVariableOpв&model_1/dense_5/BiasAdd/ReadVariableOpв%model_1/dense_5/MatMul/ReadVariableOpв"model_1/embedding/embedding_lookupвLmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2u
"model_1/tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Л
model_1/tf.reshape_1/ReshapeReshapetext_xf+model_1/tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         Б
&model_1/text_vectorization/StringLowerStringLower%model_1/tf.reshape_1/Reshape:output:0*#
_output_shapes
:         р
-model_1/text_vectorization/StaticRegexReplaceStaticRegexReplace/model_1/text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite m
,model_1/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B Є
4model_1/text_vectorization/StringSplit/StringSplitV2StringSplitV26model_1/text_vectorization/StaticRegexReplace:output:05model_1/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Л
:model_1/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Н
<model_1/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Н
<model_1/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╠
4model_1/text_vectorization/StringSplit/strided_sliceStridedSlice>model_1/text_vectorization/StringSplit/StringSplitV2:indices:0Cmodel_1/text_vectorization/StringSplit/strided_slice/stack:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_1:output:0Emodel_1/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskЖ
<model_1/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: И
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:И
>model_1/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
6model_1/text_vectorization/StringSplit/strided_slice_1StridedSlice<model_1/text_vectorization/StringSplit/StringSplitV2:shape:0Emodel_1/text_vectorization/StringSplit/strided_slice_1/stack:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Gmodel_1/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskс
]model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast=model_1/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╪
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast?model_1/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: °
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:▒
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: є
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdpmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: н
kmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : №
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateromodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0tmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: Н
fmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastmmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: │
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ф
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: й
gmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :ё
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2nmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0pmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ф
emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuljmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: х
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: щ
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumcmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: м
imodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ъ
jmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountamodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0rmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         ж
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ∙
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumqmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         ▓
hmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ж
dmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
_model_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2qmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0emodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0mmodel_1/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         У
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Ymodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle=model_1/text_vectorization/StringSplit/StringSplitV2:values:0Zmodel_1_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╓
.model_1/text_vectorization/string_lookup/EqualEqual=model_1/text_vectorization/StringSplit/StringSplitV2:values:00model_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         л
1model_1/text_vectorization/string_lookup/SelectV2SelectV22model_1/text_vectorization/string_lookup/Equal:z:03model_1_text_vectorization_string_lookup_selectv2_tUmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         з
1model_1/text_vectorization/string_lookup/IdentityIdentity:model_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         y
7model_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R И
/model_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       ╢
>model_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor8model_1/text_vectorization/RaggedToTensor/Const:output:0:model_1/text_vectorization/string_lookup/Identity:output:0@model_1/text_vectorization/RaggedToTensor/default_value:output:0?model_1/text_vectorization/StringSplit/strided_slice_1:output:0=model_1/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSо
"model_1/embedding/embedding_lookupResourceGather(model_1_embedding_embedding_lookup_20182Gmodel_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*;
_class1
/-loc:@model_1/embedding/embedding_lookup/20182*+
_output_shapes
:         d*
dtype0╫
+model_1/embedding/embedding_lookup/IdentityIdentity+model_1/embedding/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model_1/embedding/embedding_lookup/20182*+
_output_shapes
:         dе
-model_1/embedding/embedding_lookup/Identity_1Identity4model_1/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         d{
9model_1/global_average_pooling1d_1/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :▌
'model_1/global_average_pooling1d_1/MeanMean6model_1/embedding/embedding_lookup/Identity_1:output:0Bmodel_1/global_average_pooling1d_1/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         Х
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	└*
dtype0┤
model_1/dense_3/MatMulMatMul0model_1/global_average_pooling1d_1/Mean:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └У
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:└*
dtype0з
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └q
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         └Ц
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0ж
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АУ
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0з
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аq
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         АХ
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0е
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Т
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ж
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
model_1/dense_5/SoftmaxSoftmax model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         p
IdentityIdentity!model_1/dense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         н
NoOpNoOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp#^model_1/embedding/embedding_lookupM^model_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2H
"model_1/embedding/embedding_lookup"model_1/embedding/embedding_lookup2Ь
Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Lmodel_1/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:         
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
еn
е
B__inference_model_1_layer_call_and_return_conditional_losses_20800
text_xfU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_20780:	РN 
dense_3_20784:	└
dense_3_20786:	└!
dense_4_20789:
└А
dense_4_20791:	А 
dense_5_20794:	А
dense_5_20796:
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвDtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         {
tf.reshape_1/ReshapeReshapetext_xf#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:         ╨
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ┌
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Г
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Е
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╤
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╚
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ш
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:й
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: █
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: е
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ф
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ¤
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: л
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: б
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :┘
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ═
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ╤
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: д
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ╩
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : с
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         к
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         є
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╛
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         Л
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         Ч
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R А
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       Ж
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSЫ
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_20780*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20383 
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_20221Ч
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_3_20784dense_3_20786*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_20399М
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_20789dense_4_20791*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_20416Л
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_20794dense_5_20796*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_20433w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ч
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2М
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:P L
'
_output_shapes
:         
!
_user_specified_name	text_xf:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
'__inference_dense_3_layer_call_fn_21058

inputs
unknown:	└
	unknown_0:	└
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_20399p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         └`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Р
ў
__inference_<lambda>_212238
4key_value_init15763_lookuptableimportv2_table_handle0
,key_value_init15763_lookuptableimportv2_keys2
.key_value_init15763_lookuptableimportv2_values	
identityИв'key_value_init15763/LookupTableImportV2 
'key_value_init15763/LookupTableImportV2LookupTableImportV24key_value_init15763_lookuptableimportv2_table_handle,key_value_init15763_lookuptableimportv2_keys.key_value_init15763_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: p
NoOpNoOp(^key_value_init15763/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :ОN:ОN2R
'key_value_init15763/LookupTableImportV2'key_value_init15763/LookupTableImportV2:!

_output_shapes	
:ОN:!

_output_shapes	
:ОN
в
~
)__inference_embedding_layer_call_fn_21029

inputs	
unknown:	РN
identityИвStatefulPartitionedCall╨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20383s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:         d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ъ
,
__inference__destroyer_21188
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
б

ї
B__inference_dense_3_layer_call_and_return_conditional_losses_21069

inputs1
matmul_readvariableop_resource:	└.
biasadd_readvariableop_resource:	└
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	└*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:└*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         └Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         └b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         └w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы
*
__inference_<lambda>_21228
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
в

Ї
B__inference_dense_5_layer_call_and_return_conditional_losses_21109

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
─E
┴
__inference__traced_save_21392
file_prefix3
/savev2_embedding_embeddings_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const_10

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: є
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*Ь
valueТBП!B:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHп
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ж
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const_10"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!		Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ў
_input_shapesф
с: :	РN:	└:└:
└А:А:	А:: : : : : ::: : : : :	РN:	└:└:
└А:А:	А::	РN:	└:└:
└А:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	РN:%!

_output_shapes
:	└:!

_output_shapes	
:└:&"
 
_output_shapes
:
└А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	РN:%!

_output_shapes
:	└:!

_output_shapes	
:└:&"
 
_output_shapes
:
└А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::%!

_output_shapes
:	РN:%!

_output_shapes
:	└:!

_output_shapes	
:└:&"
 
_output_shapes
:
└А:!

_output_shapes	
:А:%!

_output_shapes
:	А:  

_output_shapes
::!

_output_shapes
: 
Ъ
,
__inference__destroyer_19861
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
е

Ў
B__inference_dense_4_layer_call_and_return_conditional_losses_20416

inputs2
matmul_readvariableop_resource:
└А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         └: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
вn
д
B__inference_model_1_layer_call_and_return_conditional_losses_20440

inputsU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
embedding_20384:	РN 
dense_3_20400:	└
dense_3_20402:	└!
dense_4_20417:
└А
dense_4_20419:	А 
dense_5_20434:	А
dense_5_20436:
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвDtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2m
tf.reshape_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         z
tf.reshape_1/ReshapeReshapeinputs#tf.reshape_1/Reshape/shape:output:0*
T0*#
_output_shapes
:         q
text_vectorization/StringLowerStringLowertf.reshape_1/Reshape:output:0*#
_output_shapes
:         ╨
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*#
_output_shapes
:         *6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ┌
,text_vectorization/StringSplit/StringSplitV2StringSplitV2.text_vectorization/StaticRegexReplace:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:         :         :Г
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Е
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Е
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      д
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:√
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╤
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         ╚
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ш
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:й
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: █
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: е
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ф
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ¤
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: л
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: б
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :┘
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ╠
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ═
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ╤
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: д
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ╩
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:         Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : с
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         к
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Ю
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         є
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:         ╛
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:         Л
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:         Ч
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:         q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R А
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"        d       Ж
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:         d*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSЫ
!embedding/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0embedding_20384*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_20383 
*global_average_pooling1d_1/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *^
fYRW
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_20221Ч
dense_3/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_1/PartitionedCall:output:0dense_3_20400dense_3_20402*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_20399М
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_20417dense_4_20419*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_20416Л
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_20434dense_5_20436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_20433w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ч
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall"^embedding/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2М
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╡
┌
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_21155
inputs_text
unknown	
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3	
identityИвStatefulPartitionedCall@
ShapeShapeinputs_text*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskB
Shape_1Shapeinputs_text*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:L
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB B l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:         *
dtype0*
shape:         щ
StatefulPartitionedCallStatefulPartitionedCallPlaceholderWithDefault:output:0inputs_textunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
	2				*
Tout
2	*:
_output_shapes(
&:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *!
fR
__inference_pruned_19889o
IdentityIdentity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_nameinputs/text:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "█L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_defaultЧ
9
examples-
serving_default_examples:0         >
output_02
StatefulPartitionedCall_3:0         tensorflow/serving/predict2K

asset_path_initializer:0-vocab_compute_and_apply_vocabulary_vocabulary2M

asset_path_initializer_1:0-vocab_compute_and_apply_vocabulary_vocabulary:Щй
щ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

	optimizer
		tft_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
P
_lookup_layer
	keras_api
_adapt_function"
_tf_keras_layer
╡

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
е
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
╦
$< _saved_model_loader_tracked_dict
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_model
▀
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratemЦ$mЧ%mШ,mЩ-mЪ4mЫ5mЬvЭ$vЮ%vЯ,vа-vб4vв5vг"
	optimizer
Q
1
$2
%3
,4
-5
46
57"
trackable_list_wrapper
Q
0
$1
%2
,3
-4
45
56"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ъ2ч
'__inference_model_1_layer_call_fn_20465
'__inference_model_1_layer_call_fn_20833
'__inference_model_1_layer_call_fn_20860
'__inference_model_1_layer_call_fn_20656└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╓2╙
B__inference_model_1_layer_call_and_return_conditional_losses_20941
B__inference_model_1_layer_call_and_return_conditional_losses_21022
B__inference_model_1_layer_call_and_return_conditional_losses_20728
B__inference_model_1_layer_call_and_return_conditional_losses_20800└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╦B╚
 __inference__wrapped_model_20211text_xf"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
Mserving_default"
signature_map
"
_generic_user_object
L
Nlookup_table
Otoken_counts
P	keras_api"
_tf_keras_layer
"
_generic_user_object
╛2╗
__inference_adapt_step_18554Ъ
У▓П
FullArgSpec
argsЪ

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
':%	РN2embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╙2╨
)__inference_embedding_layer_call_fn_21029в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_embedding_layer_call_and_return_conditional_losses_21038в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ё2ю
:__inference_global_average_pooling1d_1_layer_call_fn_21043п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
М2Й
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_21049п
ж▓в
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
!:	└2dense_3/kernel
:└2dense_3/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
╤2╬
'__inference_dense_3_layer_call_fn_21058в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_3_layer_call_and_return_conditional_losses_21069в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
": 
└А2dense_4/kernel
:А2dense_4/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
╤2╬
'__inference_dense_4_layer_call_fn_21078в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_4_layer_call_and_return_conditional_losses_21089в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
!:	А2dense_5/kernel
:2dense_5/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
н
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
╤2╬
'__inference_dense_5_layer_call_fn_21098в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_5_layer_call_and_return_conditional_losses_21109в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Т
j	_imported
k_structured_inputs
l_structured_outputs
m_output_to_inputs_map
n_wrapped_function"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Ь2Щ
8__inference_transform_features_layer_layer_call_fn_20272
8__inference_transform_features_layer_layer_call_fn_21124в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_21155
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_20318в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╦B╚
#__inference_signature_wrapper_20129examples"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
j
v_initializer
w_create_resource
x_initialize
y_destroy_resourceR jCustom.StaticHashTable
Q
z_create_resource
{_initialize
|_destroy_resourceR Z
tableде
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
─
}created_variables
~	resources
trackable_objects
Аinitializers
Бassets
В
signatures
$Г_self_saveable_object_factories
ntransform_fn"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
0B.
__inference_pruned_19889inputsinputs_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Дtotal

Еcount
Ж	variables
З	keras_api"
_tf_keras_metric
c

Иtotal

Йcount
К
_fn_kwargs
Л	variables
М	keras_api"
_tf_keras_metric
"
_generic_user_object
▒2о
__inference__creator_21160П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡2▓
__inference__initializer_21168П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
│2░
__inference__destroyer_21173П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▒2о
__inference__creator_21178П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡2▓
__inference__initializer_21183П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
│2░
__inference__destroyer_21188П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
 "
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
О0"
trackable_list_wrapper
(
П0"
trackable_list_wrapper
-
Рserving_default"
signature_map
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
Д0
Е1"
trackable_list_wrapper
.
Ж	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
И0
Й1"
trackable_list_wrapper
.
Л	variables"
_generic_user_object
V
О_initializer
С_create_resource
Т_initialize
У_destroy_resourceR 
T
Ф	_filename
$Х_self_saveable_object_factories"
_generic_user_object
* 
╤B╬
#__inference_signature_wrapper_19902inputsinputs_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▒2о
__inference__creator_19938П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡2▓
__inference__initializer_19867П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
│2░
__inference__destroyer_19861П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
*
 "
trackable_dict_wrapper
,:*	РN2Adam/embedding/embeddings/m
&:$	└2Adam/dense_3/kernel/m
 :└2Adam/dense_3/bias/m
':%
└А2Adam/dense_4/kernel/m
 :А2Adam/dense_4/bias/m
&:$	А2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
,:*	РN2Adam/embedding/embeddings/v
&:$	└2Adam/dense_3/kernel/v
 :└2Adam/dense_3/bias/v
':%
└А2Adam/dense_4/kernel/v
 :А2Adam/dense_4/bias/v
&:$	А2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
▌B┌
__inference_save_fn_21207checkpoint_key"к
Щ▓Х
FullArgSpec
argsЪ
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в	
К 
ГBА
__inference_restore_fn_21215restored_tensors_0restored_tensors_1"╡
Ч▓У
FullArgSpec
argsЪ 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в
	К
	К	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_96
__inference__creator_19938в

в 
к "К 6
__inference__creator_21160в

в 
к "К 6
__inference__creator_21178в

в 
к "К 8
__inference__destroyer_19861в

в 
к "К 8
__inference__destroyer_21173в

в 
к "К 8
__inference__destroyer_21188в

в 
к "К @
__inference__initializer_19867ФНв

в 
к "К A
__inference__initializer_21168Nопв

в 
к "К :
__inference__initializer_21183в

в 
к "К Щ
 __inference__wrapped_model_20211uNжзи$%,-450в-
&в#
!К
text_xf         
к "1к.
,
dense_5!К
dense_5         j
__inference_adapt_step_18554JOй?в<
5в2
0Т-в
К         IteratorSpec 
к "
 г
B__inference_dense_3_layer_call_and_return_conditional_losses_21069]$%/в,
%в"
 К
inputs         
к "&в#
К
0         └
Ъ {
'__inference_dense_3_layer_call_fn_21058P$%/в,
%в"
 К
inputs         
к "К         └д
B__inference_dense_4_layer_call_and_return_conditional_losses_21089^,-0в-
&в#
!К
inputs         └
к "&в#
К
0         А
Ъ |
'__inference_dense_4_layer_call_fn_21078Q,-0в-
&в#
!К
inputs         └
к "К         Аг
B__inference_dense_5_layer_call_and_return_conditional_losses_21109]450в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ {
'__inference_dense_5_layer_call_fn_21098P450в-
&в#
!К
inputs         А
к "К         з
D__inference_embedding_layer_call_and_return_conditional_losses_21038_/в,
%в"
 К
inputs         d	
к ")в&
К
0         d
Ъ 
)__inference_embedding_layer_call_fn_21029R/в,
%в"
 К
inputs         d	
к "К         d╘
U__inference_global_average_pooling1d_1_layer_call_and_return_conditional_losses_21049{IвF
?в<
6К3
inputs'                           

 
к ".в+
$К!
0                  
Ъ м
:__inference_global_average_pooling1d_1_layer_call_fn_21043nIвF
?в<
6К3
inputs'                           

 
к "!К                  ╖
B__inference_model_1_layer_call_and_return_conditional_losses_20728qNжзи$%,-458в5
.в+
!К
text_xf         
p 

 
к "%в"
К
0         
Ъ ╖
B__inference_model_1_layer_call_and_return_conditional_losses_20800qNжзи$%,-458в5
.в+
!К
text_xf         
p

 
к "%в"
К
0         
Ъ ╢
B__inference_model_1_layer_call_and_return_conditional_losses_20941pNжзи$%,-457в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ ╢
B__inference_model_1_layer_call_and_return_conditional_losses_21022pNжзи$%,-457в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ П
'__inference_model_1_layer_call_fn_20465dNжзи$%,-458в5
.в+
!К
text_xf         
p 

 
к "К         П
'__inference_model_1_layer_call_fn_20656dNжзи$%,-458в5
.в+
!К
text_xf         
p

 
к "К         О
'__inference_model_1_layer_call_fn_20833cNжзи$%,-457в4
-в*
 К
inputs         
p 

 
к "К         О
'__inference_model_1_layer_call_fn_20860cNжзи$%,-457в4
-в*
 К
inputs         
p

 
к "К         М
__inference_pruned_19889я
клНмнxвu
nвk
iкf
5
category)К&
inputs/category         
-
text%К"
inputs/text         
к "gкd
4
category_xf%К"
category_xf         	
,
text_xf!К
text_xf         y
__inference_restore_fn_21215YOKвH
Aв>
К
restored_tensors_0
К
restored_tensors_1	
к "К Ф
__inference_save_fn_21207ЎO&в#
в
К
checkpoint_key 
к "╚Ъ─
`к]

nameК
0/name 
#

slice_specК
0/slice_spec 

tensorК
0/tensor
`к]

nameК
1/name 
#

slice_specК
1/slice_spec 

tensorК
1/tensor	И
#__inference_signature_wrapper_19902р
клНмнiвf
в 
_к\
*
inputs К
inputs         
.
inputs_1"К
inputs_1         "gкd
4
category_xf%К"
category_xf         	
,
text_xf!К
text_xf         ▓
#__inference_signature_wrapper_20129КклНмнNжзи$%,-459в6
в 
/к,
*
examplesК
examples         "3к0
.
output_0"К
output_0         ▀
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_20318З
клНмн:в7
0в-
+к(
&
textК
text         
к "=в:
3к0
.
text_xf#К 
	0/text_xf         
Ъ ц
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_21155О
клНмнAв>
7в4
2к/
-
text%К"
inputs/text         
к "=в:
3к0
.
text_xf#К 
	0/text_xf         
Ъ ╖
8__inference_transform_features_layer_layer_call_fn_20272{
клНмн:в7
0в-
+к(
&
textК
text         
к "1к.
,
text_xf!К
text_xf         ┐
8__inference_transform_features_layer_layer_call_fn_21124В
клНмнAв>
7в4
2к/
-
text%К"
inputs/text         
к "1к.
,
text_xf!К
text_xf         