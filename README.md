<h3 align="center">
  <img src="assets/text_predictor_icon_web.png" width="300">
</h3>

# Text Predictor
Character-level **RNN** (Recurrent Neural Net) **LSTM** (Long Short-Term Memory) implemented in Python 2.7/TensorFlow in order to predict a text based on a given dataset. 

<br>

Check out corresponding Medium article:

[Text Predictor - Generating Rap Lyrics with Recurrent Neural Networks (LSTMs)📄](https://towardsdatascience.com/text-predictor-generating-rap-lyrics-with-recurrent-neural-networks-lstms-c3a1acbbda79)

---

Heavily influenced by: [http://karpathy.github.io/2015/05/21/rnn-effectiveness/]().

## Idea
1. Train RNN LSTM  on a given dataset (.txt file).
2. Predict text based on a trained model.

## Datasets
	kanye - Kanye West's discography (332 KB)
	darwin - the complete works of Charles Darwin (20 MB)
	reuters - a collection of Reuters headlines (95 MB)
	war_and_peace - Leo Tolstoy's War and Peace novel (3 MB)
	wikipedia - excerpt from English Wikipedia (48 MB) 
	hackernews - a collection of Hackernews headlines (90 KB)
	sherlock - a collection of books with Sherlock Holmes (3 MB)
	shakespeare - the complete works of William Shakespeare (4 MB)
Feel free to add new datasets. Just create a folder in the `./data` directory and put an `input.txt` file there. Output file along with the training plot will be automatically generated there.
	
	
## Usage
1. Clone the repo.
2. Go to the project's root folder.
3. Install required packages`pip install -r requirements.txt`.
4. `python text_predictor.py <dataset>`.


## Results

Each dataset were trained with the same hyperparameters.

**Hyperparameters**

	BATCH_SIZE = 32
	SEQUENCE_LENGTH = 50
	LEARNING_RATE = 0.01
	DECAY_RATE = 0.97
	HIDDEN_LAYER_SIZE = 256
	CELLS_SIZE = 2



### Sherlock
<img src="data/sherlock/loss.png" width="500">

Iteration: **0**

	 l é°£I." r, iEgPylXyg
	m .iüTû  Ccy2M]zTâ.  sSRM£t é5 ’îRlT QAlY4Kv"é)kP£Str5/lQVu )Pe0/;s8leJ.£m40tîJîwB`0]½jyûA`BJi'omNx½2zG iH:gqri76b&g)ie18PM£vA7pßKâNQ6
	2 û?]wg£Jo4qCde,’.'G,h &wIUaDuîxq`cqb!kf5yB

<br>

Iteration: **500**

	"Other. I
	     unwallfore of his had Sommopilor out he hase you thed I it.
	
	     Book into here, but I told at ht it something do was sack knet afminture-ly. We moke, do oR before drinessast farm. I
 

<br>

Iteration: **1000**

   	some to see me tignaius
     rely."
	
     There that you'd them were from I
     should not have any take an watchate save now out," said Hodden?"
	
     "Th, a lott remarks. Showed."
	
     "A joan?"
     
     
<br>

Iteration: **100000**

 	Then mention.""Quite
     I gather is stillar in silence was written on the whom I reward an
     details grieves of his east back. The week shook this strength.
     There was no mystery for y

<br>

### Hackernews
<img src="data/hackernews/loss.png" width="500">

Iteration: **0**

	 %‘l~E4*1[▲)j&”&T$b’]u:…–.2WPUlFLu*)Eµk`qb€[QoE'aLesP‘U4.q
	o_Z2ZPGé‘MIn8beXSB=B“dNuy…uµ20P8vL”(#
	-`H/€€:–mµ,g+WU5'^cA=Y–t
	z+.I,—6N7?f;7Z)nk
	i≠?YsW"iHJ77€Ty y_eS5pnwN6‘
	%oVhkXr[xAlc*Tx’S1–J1LlHN'SuHEsiH

<br>

Iteration: **500**

	 us codhy in N. DeveloLenic pare abouts scax
	Microsign Sy Scodwars
	Machons Startians: The is abandied
	Payer Progroads Procinters
	How 1.05)
	Trase elacts Macasications Data Freit Paily trigha bourni

<br>

Iteration: **1000**

	 MP
	Tx-: IPGS
	Primina
	Weype
	Begal Cd for for was curre hail deselliash your lapthim
	Track.L
	Tvist
	Ubunts writing the like review
	Swifch, Now internet will Net 10 TS some libission
	Lass and dom
	
<br>

Iteration: **100000**

	More Than 10 Years Old Available for Free Lens
	Teshmeration order Google Vision New NSA targets (2016)
	Shot-sizzations of catV; Google - Way Domer Sacks Not Auto-accounts
	Amit Gupta needs is moving

<br>

### Shakespeare
<img src="data/shakespeare/loss.png" width="500">

Iteration: **0**

	TfzVRzdYlDehaDHIhzEiZ&,3knZtHJD]kBOFCpWH.wkWCDVHAK;JcoOMpHJtVNvpcrRSZ,hccUNQ EyG -kpEuvR;MW[JWm;EWv]Au!]EIriywVeGYdljvLkoFMRdikQV:AyoSij.M.;R'lK
	vdtnVkxtzL!'qtW$emHfStGUOoK;LJ h
	LSyL ?P$KET Z?muR$reIB

<br>

Iteration: **500**

	ticlother them his steaks? whom father-ple plaise't!
	
	HORATIO:
	
	GLOILUS:
	Le wime heast,
	'Tind soul a bear if thy Gulithes? Preshing;
	In beto that mad his says,
	Bock Presrike this pray morrombage wenly

<br>

Iteration: **1000**

	HENI:
	If which fout in must likest part sors and merr'd?
	E sin even and mel full and gooder?
	
	BRUTUS:
	Heno Egison to a puenbiloot vieter.
	
	DROMIO OF SYRACUSE:
	That is
	never standshruced meledder morng
	
<br>

Iteration: **100000**

	Be feast, tent?
	
	LYSANDER:
	And thou love so kiss, to dipate.
	
	All Cornasiers of Atheniansiage are to my sake; but where in end.
	
	APEMANTUS:
	Did such a pays. Go, we'll proof.
	
	BERTRAM:
	I am reason'dst 
	
<br>

### War and Peace
<img src="data/war_and_peace/loss.png" width="500">

Iteration: **0**

	oeLêQ8r2),*FV00Krj﻿F':=BEYGêW﻿f1
	d'qwAd,X,m;à8)j9V)ExSRaox!l(=3étQäsHOlUZ
	YgDFI/mpF
	J﻿P.A7W)5bqN,iC àAiiGp, R﻿k-v1Qm:9ZoX*qDJwq,BW!:59tNv?êR"aE﻿1M;snov=:rlK *oFxK2mL,6V5br﻿Q9LN*LwXGe2dpo3C?mx=i)rYr=f9

<br>

Iteration: **500**

	un-
	more-alre depiw.
	
	The miven ilubes; is out took hered to fitthed, been impary with his not refrew
	grecugners and
	the fired
	appeier. On; was expring. Gche wast.
	
	
	Himpery
	at it of been th

<br>

Iteration: **1000**

	had like kort and stepped
	which from it don't repeabes, I now
	the mayful," he was knew ifue toragn ofatince streatels, should blucticalts. Peterning letter, they his voice went the ninding
	sonison 

<br>

Iteration: **100000**

	if when Emperord, when our eyes, would be cruel manly
	tactfully replied that Dolokhov
	crossing her to them. He looked his face in snow face, but sound at closely deigning dogron (for Germans: "We le

<br>

### Darwin
<img src="data/darwin/loss.png" width="500">


Iteration: **0**

	W⅝[—¤Sé½,°Rá{ú⅛ιW‘œΠ┘NfnáχRœ|NE~{A┐!àμ£μvk¤⅜%àιW⅜,—E.lJW⅓VQαÉIl—
	á¹′(œM⅜sOΠ¹┘+öô,vt(ë†XYœα^aφIyôdCAι8⅞”¼┐Pü+wœ[N)3⅞(ςÜZçàôeφe⅞–bz⅝dε5É<6D;…T|Qχ…o,z %&T′x=“ΧÂ£×ιD&“Bî·…*—νKt1dHaùuÈ;w*[┘}§Uï(r¾rω&œ”–¹C

<br>

Iteration: **500**

	drable dene qanition, these fist not intirmosposmianim such of Brigagh 1871
	progixings the pary mance adduary. The litter mame is for the
	amber not notnot the digracke.  If a amy inter of sindenly u

<br>

Iteration: **1000**

	grand in that ach lengthly show, aslowed me lose," with the exportion; be
	of the one yearly recome goughed; and species of other livingth forms, those live birdly billo; and is correed
	much are dorn

<br>

Iteration: **100000**

	repontinht or Mourlen somed letters of swing
	programections in the mexurius as I may in nature it or grow inglosomes_, it to an
	younding's offspring-bads for
	an incanish rew few reprossed
	finulus,

<br>

### Kanye
<img src="data/kanye/loss.png" width="500">

Iteration: **0**

	9hu71JQ)eA"oqwrAAUwG5Wv7rvM60[*$Y!:1v*8tbkB+k 8IGn)QWv8NR.Spi3BtK[VteRer1GQ,it"kD?XVel3lNuN+G//rI' Sl?ssm
	 NbH # Yk2uY"fmSVFah(B]uYZv+2]nsMX(qX9s+Rn+YAM.y/2 Hp9a,ZQOu,dM3.;im$Jca4E6(HS'D
	[itYYQG#(gahU(gGoFYi)ucubL3 #iU32 8rdwIG7HJYSpDG*j,5
	4phPY'SqiZMpVH-[KEkUjNFyIC#AInX
	ys0sw8&IaNC1mYSs$*lW#6e,X(aJDgtx"!u-*N6J(N&Awk7X3P0nWvx)oJLVbWncCS
	] P2wQTKTtSXrK9pjR0x5bcwU$ KA7"y+ :0:?wd(BOX1:,LICy]-v/)Y5K(G.Sa qP1vf(LXUDe4jqU3a3s$!cxVv(TO#yRoiXD#ZXw0ny09lu;gFaIqCiyEB)YhP,P
	#G$T/].X3m]b9fc
	hgsn.QG2WIZ3JS#I

<br>

Iteration: **1000**

	am our 200 shought 2 and but
	One we -fuckister do fresh smandles
	Juco pick with to sont party agmagle
	Then I no meant he don't ganiscimes mad is so cametie want
	What
	Mama sumin' find Abortsimes, man
	You's partystend to heed)
	Never)
	Whats what a gonna bodry Find down
	Wihe a mostry that day to the news winces
	(Had what icherced and I'm nigga"" and some talk to beinn shood late you, fly Me down
	Youce, I and fleassy is
	
<br>

Iteration: **10000**

	as the comphol of step
	Stand American, no more
	Yeah my Benz,.AD and brosi?
	Cause you'll take me, breaks to the good I'll never said, ""I met her bitch's pussy is a proll ...
	WHO WILL Say everything
	We been a minute it's liberatimes?
	(Stop that religious and the hegasn of me, steps dead)
	I can't contlights you
	I bet stop me, I won't you
	I cant face and flesed
	Tellin' it and sales there
	Got a niggas ass a lots over?
	So I clay messin 6 wrong baby
	Dog, we lose, ""Can't say how I'm heren
	
<br>

Iteration: **231000**
	
	right here, history on you
	Dees so can do now, sippin' with niggas want to go
	
	[Hook]
	Good morning!
	He wanna kend care helped all wing… the live, man
	I'm taking all in my sleep, Im out him and I ain't inspired?
	Okay, go you're pastor save being make them
	White hit Victure up, it can go down
	
	[Outro: Kanye West]
	One time
	To make them other you're like Common
	A lit it, I'mma bridgeidenace before the most high
	Ugh! we get much higher
