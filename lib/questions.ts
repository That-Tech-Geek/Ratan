export type Question = {
  id: string;
  class: 8 | 9;
  subject: "maths" | "science";
  topic: string;
  subtopic: string;
  difficulty: "foundation" | "core" | "stretch";
  prompt: string;
  options: string[];
  answerIndex: number;
  explanationOdia: string;
  reviewStatus: "teacher-approved";
};

const M8: Question[] = [
 {id:"M8-NUM-001",class:8,subject:"maths",topic:"Numbers",subtopic:"fractions",difficulty:"foundation",prompt:"What is 3/4 + 1/4?",options:["1","3/8","4/8","2"],answerIndex:0,explanationOdia:"3/4 ଓ 1/4 ମିଶି 4/4, ଅର୍ଥାତ୍ 1 ହୁଏ।",reviewStatus:"teacher-approved"},
 {id:"M8-NUM-002",class:8,subject:"maths",topic:"Numbers",subtopic:"rational numbers",difficulty:"core",prompt:"Which number is greatest?",options:["-2","-1/2","0","-3"],answerIndex:2,explanationOdia:"0 ସମସ୍ତ ଋଣାତ୍ମକ ସଂଖ୍ୟାଠାରୁ ବଡ଼।",reviewStatus:"teacher-approved"},
 {id:"M8-ALG-001",class:8,subject:"maths",topic:"Algebra",subtopic:"linear expressions",difficulty:"foundation",prompt:"If x + 7 = 12, what is x?",options:["3","5","7","19"],answerIndex:1,explanationOdia:"ଦୁଇ ପାର୍ଶ୍ୱରୁ 7 ବିୟୋଗ କଲେ x = 5।",reviewStatus:"teacher-approved"},
 {id:"M8-ALG-002",class:8,subject:"maths",topic:"Algebra",subtopic:"identities",difficulty:"core",prompt:"What is 5a + 2a?",options:["7","7a","10a","a⁷"],answerIndex:1,explanationOdia:"ସମାନ ପଦ 5a ଓ 2a ଯୋଗ କଲେ 7a।",reviewStatus:"teacher-approved"},
 {id:"M8-GEO-001",class:8,subject:"maths",topic:"Geometry",subtopic:"angles",difficulty:"foundation",prompt:"The angles of a triangle add up to:",options:["90°","180°","270°","360°"],answerIndex:1,explanationOdia:"ତ୍ରିଭୁଜର ତିନୋଟି ଅନ୍ତଃକୋଣର ଯୋଗ 180°।",reviewStatus:"teacher-approved"},
 {id:"M8-GEO-002",class:8,subject:"maths",topic:"Geometry",subtopic:"quadrilaterals",difficulty:"core",prompt:"A square has how many lines of symmetry?",options:["1","2","3","4"],answerIndex:3,explanationOdia:"ବର୍ଗର 4ଟି ସମମିତି ରେଖା ଅଛି।",reviewStatus:"teacher-approved"},
 {id:"M8-MSR-001",class:8,subject:"maths",topic:"Mensuration",subtopic:"area",difficulty:"foundation",prompt:"What is the area of a rectangle 6 cm long and 4 cm wide?",options:["10 cm²","20 cm²","24 cm²","48 cm²"],answerIndex:2,explanationOdia:"କ୍ଷେତ୍ରଫଳ = ଲମ୍ବ × ପ୍ରସ୍ଥ = 6 × 4 = 24 cm²।",reviewStatus:"teacher-approved"},
 {id:"M8-RAT-001",class:8,subject:"maths",topic:"Ratio",subtopic:"proportion",difficulty:"core",prompt:"If 2 pens cost ₹20, how much do 5 pens cost at the same rate?",options:["₹30","₹40","₹50","₹60"],answerIndex:2,explanationOdia:"ପ୍ରତି ପେନର ମୂଲ୍ୟ ₹10, ତେଣୁ 5ଟିର ମୂଲ୍ୟ ₹50।",reviewStatus:"teacher-approved"},
 {id:"M8-DAT-001",class:8,subject:"maths",topic:"Data",subtopic:"mean",difficulty:"core",prompt:"What is the mean of 4, 6 and 8?",options:["5","6","7","18"],answerIndex:1,explanationOdia:"ଯୋଗ 18; 3ରେ ଭାଗ କଲେ 6।",reviewStatus:"teacher-approved"},
 {id:"M8-PROB-001",class:8,subject:"maths",topic:"Probability",subtopic:"simple probability",difficulty:"stretch",prompt:"A fair die is rolled once. What is the probability of getting 6?",options:["1/2","1/3","1/6","5/6"],answerIndex:2,explanationOdia:"6ଟି ସମ୍ଭାବ୍ୟ ଫଳରୁ 1ଟି ଫଳ 6, ତେଣୁ ସମ୍ଭାବନା 1/6।",reviewStatus:"teacher-approved"}
];

const S8: Question[] = [
 {id:"S8-CROP-001",class:8,subject:"science",topic:"Crop Production",subtopic:"agricultural practices",difficulty:"foundation",prompt:"Which tool is commonly used for loosening soil?",options:["Plough","Thermometer","Compass","Stethoscope"],answerIndex:0,explanationOdia:"ମାଟିକୁ ଢିଲା କରିବା ପାଇଁ ଲଙ୍ଗଳ ବ୍ୟବହାର କରାଯାଏ।",reviewStatus:"teacher-approved"},
 {id:"S8-MICRO-001",class:8,subject:"science",topic:"Microorganisms",subtopic:"useful microbes",difficulty:"core",prompt:"Which microorganism is used to make curd?",options:["Lactobacillus","Plasmodium","Amoeba","Rhizobium"],answerIndex:0,explanationOdia:"Lactobacillus ଦୁଧକୁ ଦହିରେ ପରିଣତ କରେ।",reviewStatus:"teacher-approved"},
 {id:"S8-SYN-001",class:8,subject:"science",topic:"Cell",subtopic:"plant cells",difficulty:"foundation",prompt:"Which structure controls many activities of a cell?",options:["Cell wall","Nucleus","Vacuole","Chloroplast"],answerIndex:1,explanationOdia:"ନ୍ୟୁକ୍ଲିଅସ୍ କୋଷର ଅନେକ କାର୍ଯ୍ୟ ନିୟନ୍ତ୍ରଣ କରେ।",reviewStatus:"teacher-approved"},
 {id:"S8-REPRO-001",class:8,subject:"science",topic:"Reproduction",subtopic:"asexual reproduction",difficulty:"core",prompt:"Budding is commonly seen in:",options:["Yeast","Mango","Fish","Frog"],answerIndex:0,explanationOdia:"Yeast ରେ budding ଦ୍ୱାରା ଅଲିଙ୍ଗିକ ପ୍ରଜନନ ହୁଏ।",reviewStatus:"teacher-approved"},
 {id:"S8-FORCE-001",class:8,subject:"science",topic:"Force and Pressure",subtopic:"force effects",difficulty:"foundation",prompt:"A push or pull on an object is called:",options:["Energy","Force","Mass","Volume"],answerIndex:1,explanationOdia:"ବସ୍ତୁ ଉପରେ ଧକ୍କା ବା ଟାଣକୁ ବଳ କୁହାଯାଏ।",reviewStatus:"teacher-approved"},
 {id:"S8-FRIC-001",class:8,subject:"science",topic:"Friction",subtopic:"everyday friction",difficulty:"core",prompt:"Which surface usually produces more friction?",options:["Smooth glass","Rough road","Ice","Oiled metal"],answerIndex:1,explanationOdia:"ଖରାପଣ ବଢ଼ିଲେ ସାଧାରଣତଃ ଘର୍ଷଣ ବଢ଼େ।",reviewStatus:"teacher-approved"},
 {id:"S8-SOUND-001",class:8,subject:"science",topic:"Sound",subtopic:"vibration",difficulty:"foundation",prompt:"Sound is produced by:",options:["Vibrations","Gravity","Light","Magnetism"],answerIndex:0,explanationOdia:"କମ୍ପନ କରୁଥିବା ବସ୍ତୁରୁ ଶବ୍ଦ ଉତ୍ପନ୍ନ ହୁଏ।",reviewStatus:"teacher-approved"},
 {id:"S8-LIGHT-001",class:8,subject:"science",topic:"Light",subtopic:"reflection",difficulty:"core",prompt:"The bouncing back of light from a surface is called:",options:["Refraction","Reflection","Absorption","Dispersion"],answerIndex:1,explanationOdia:"ଆଲୋକ ପୃଷ୍ଠରୁ ଫେରିଆସିବାକୁ ପ୍ରତିଫଳନ କୁହାଯାଏ।",reviewStatus:"teacher-approved"},
 {id:"S8-CHEM-001",class:8,subject:"science",topic:"Chemical Effects",subtopic:"conductivity",difficulty:"stretch",prompt:"Which liquid is most likely to conduct electricity?",options:["Distilled water","Salt solution","Cooking oil","Pure kerosene"],answerIndex:1,explanationOdia:"ଲୁଣ ଦ୍ରବଣରେ ଆୟନ ଥିବାରୁ ଏହା ବିଦ୍ୟୁତ୍ ପରିବହନ କରେ।",reviewStatus:"teacher-approved"},
 {id:"S8-CONS-001",class:8,subject:"science",topic:"Conservation",subtopic:"wildlife",difficulty:"core",prompt:"A species found only in a particular region is called:",options:["Domestic","Endemic","Migratory","Artificial"],answerIndex:1,explanationOdia:"କେବଳ ନିର୍ଦ୍ଦିଷ୍ଟ ଅଞ୍ଚଳରେ ମିଳୁଥିବା ପ୍ରଜାତିକୁ endemic କୁହାଯାଏ।",reviewStatus:"teacher-approved"}
];

const M9: Question[] = [
 {id:"M9-NUM-001",class:9,subject:"maths",topic:"Number Systems",subtopic:"irrational numbers",difficulty:"foundation",prompt:"Which is irrational?",options:["1/2","0.25","√2","4"],answerIndex:2,explanationOdia:"√2 କୁ ଦୁଇ ପୂର୍ଣ୍ଣ ସଂଖ୍ୟାର ଅନୁପାତ ଭାବେ ଲେଖିହେବ ନାହିଁ।",reviewStatus:"teacher-approved"},
 {id:"M9-POLY-001",class:9,subject:"maths",topic:"Polynomials",subtopic:"degree",difficulty:"foundation",prompt:"What is the degree of 3x² + 2x - 5?",options:["1","2","3","5"],answerIndex:1,explanationOdia:"ସର୍ବାଧିକ ଘାତ 2, ତେଣୁ degree 2।",reviewStatus:"teacher-approved"},
 {id:"M9-LINE-001",class:9,subject:"maths",topic:"Coordinate Geometry",subtopic:"coordinates",difficulty:"core",prompt:"The point (0, 5) lies on which axis?",options:["x-axis","y-axis","both","neither"],answerIndex:1,explanationOdia:"x-coordinate 0 ଥିଲେ ବିନ୍ଦୁ y-axis ଉପରେ ଥାଏ।",reviewStatus:"teacher-approved"},
 {id:"M9-LINE-002",class:9,subject:"maths",topic:"Linear Equations",subtopic:"two variables",difficulty:"core",prompt:"Which ordered pair satisfies x + y = 5?",options:["(1,1)","(2,3)","(4,2)","(5,2)"],answerIndex:1,explanationOdia:"2 + 3 = 5, ତେଣୁ (2,3) ସଠିକ୍।",reviewStatus:"teacher-approved"},
 {id:"M9-GEO-001",class:9,subject:"maths",topic:"Geometry",subtopic:"triangles",difficulty:"core",prompt:"If two sides of a triangle are equal, the angles opposite them are:",options:["Unequal","Equal","Always 90°","Always 60°"],answerIndex:1,explanationOdia:"ସମଦ୍ୱିବାହୁ ତ୍ରିଭୁଜର ସମାନ ପାର୍ଶ୍ୱ ବିପରୀତ କୋଣ ସମାନ।",reviewStatus:"teacher-approved"},
 {id:"M9-CIRC-001",class:9,subject:"maths",topic:"Circles",subtopic:"radius",difficulty:"foundation",prompt:"If a circle's diameter is 10 cm, its radius is:",options:["2 cm","5 cm","10 cm","20 cm"],answerIndex:1,explanationOdia:"ବ୍ୟାସ = 2 × ବ୍ୟାସାର୍ଦ୍ଧ, ତେଣୁ ବ୍ୟାସାର୍ଦ୍ଧ 5 cm।",reviewStatus:"teacher-approved"},
 {id:"M9-HERON-001",class:9,subject:"maths",topic:"Mensuration",subtopic:"area",difficulty:"stretch",prompt:"A triangle has sides 3 cm, 4 cm and 5 cm. Its area is:",options:["5 cm²","6 cm²","10 cm²","12 cm²"],answerIndex:1,explanationOdia:"3-4-5 ତ୍ରିଭୁଜ ସମକୋଣୀ; କ୍ଷେତ୍ରଫଳ = 1/2 × 3 × 4 = 6 cm²।",reviewStatus:"teacher-approved"},
 {id:"M9-STAT-001",class:9,subject:"maths",topic:"Statistics",subtopic:"mean",difficulty:"core",prompt:"The mean of 2, 4, 6 and 8 is:",options:["4","5","6","20"],answerIndex:1,explanationOdia:"ଯୋଗ 20; 4ରେ ଭାଗ କଲେ 5।",reviewStatus:"teacher-approved"},
 {id:"M9-PROB-001",class:9,subject:"maths",topic:"Probability",subtopic:"experimental probability",difficulty:"core",prompt:"A coin is tossed 100 times and heads appears 47 times. The experimental probability of heads is:",options:["0.47","0.50","0.53","47"],answerIndex:0,explanationOdia:"47 ÷ 100 = 0.47।",reviewStatus:"teacher-approved"},
 {id:"M9-ALG-001",class:9,subject:"maths",topic:"Algebra",subtopic:"factorisation",difficulty:"stretch",prompt:"Which is a factorisation of x² - 9?",options:["(x-9)(x+1)","(x-3)(x+3)","(x-3)²","x(x-9)"],answerIndex:1,explanationOdia:"ଏହା difference of squares: x² - 3² = (x-3)(x+3)।",reviewStatus:"teacher-approved"}
];

const S9: Question[] = [
 {id:"S9-MAT-001",class:9,subject:"science",topic:"Matter",subtopic:"states of matter",difficulty:"foundation",prompt:"Particles in a gas are generally:",options:["Tightly packed","Far apart","Fixed in position","Motionless"],answerIndex:1,explanationOdia:"ଗ୍ୟାସର କଣିକାମାନଙ୍କ ମଧ୍ୟରେ ଅଧିକ ଖାଲି ସ୍ଥାନ ଥାଏ।",reviewStatus:"teacher-approved"},
 {id:"S9-PURE-001",class:9,subject:"science",topic:"Matter",subtopic:"mixtures",difficulty:"core",prompt:"Air is best described as a:",options:["Compound","Mixture","Element","Pure metal"],answerIndex:1,explanationOdia:"ବାୟୁରେ ଅନେକ ଗ୍ୟାସର ମିଶ୍ରଣ ରହିଛି।",reviewStatus:"teacher-approved"},
 {id:"S9-ATOM-001",class:9,subject:"science",topic:"Atoms",subtopic:"subatomic particles",difficulty:"foundation",prompt:"Which particle has a negative charge?",options:["Proton","Neutron","Electron","Nucleus"],answerIndex:2,explanationOdia:"Electron ର ଚାର୍ଜ ଋଣାତ୍ମକ।",reviewStatus:"teacher-approved"},
 {id:"S9-CELL-001",class:9,subject:"science",topic:"Cell",subtopic:"cell organelles",difficulty:"core",prompt:"Which organelle is known as the powerhouse of the cell?",options:["Nucleus","Mitochondrion","Ribosome","Vacuole"],answerIndex:1,explanationOdia:"Mitochondria କୋଷ ପାଇଁ ଶକ୍ତି ଉତ୍ପାଦନ କରେ।",reviewStatus:"teacher-approved"},
 {id:"S9-TISS-001",class:9,subject:"science",topic:"Tissues",subtopic:"plant tissues",difficulty:"core",prompt:"Which tissue transports water in plants?",options:["Phloem","Xylem","Epidermis","Meristem"],answerIndex:1,explanationOdia:"Xylem ମୂଳରୁ ଉପରକୁ ଜଳ ଓ ଖଣିଜ ବହନ କରେ।",reviewStatus:"teacher-approved"},
 {id:"S9-MOTION-001",class:9,subject:"science",topic:"Motion",subtopic:"speed",difficulty:"foundation",prompt:"A car travels 60 km in 2 hours. Its average speed is:",options:["20 km/h","30 km/h","60 km/h","120 km/h"],answerIndex:1,explanationOdia:"ବେଗ = ଦୂରତା ÷ ସମୟ = 60 ÷ 2 = 30 km/h।",reviewStatus:"teacher-approved"},
 {id:"S9-FORCE-001",class:9,subject:"science",topic:"Force",subtopic:"Newton's laws",difficulty:"core",prompt:"The SI unit of force is:",options:["Joule","Watt","Newton","Pascal"],answerIndex:2,explanationOdia:"ବଳର SI ଏକକ Newton (N)।",reviewStatus:"teacher-approved"},
 {id:"S9-GRAV-001",class:9,subject:"science",topic:"Gravitation",subtopic:"weight",difficulty:"core",prompt:"Weight depends directly on an object's:",options:["Colour","Mass and gravitational acceleration","Temperature only","Volume only"],answerIndex:1,explanationOdia:"W = mg, ତେଣୁ ଓଜନ mass ଓ gravitational acceleration ଉପରେ ନିର୍ଭର କରେ।",reviewStatus:"teacher-approved"},
 {id:"S9-WORK-001",class:9,subject:"science",topic:"Work and Energy",subtopic:"kinetic energy",difficulty:"stretch",prompt:"Energy possessed by a moving object is called:",options:["Potential energy","Kinetic energy","Chemical energy","Sound only"],answerIndex:1,explanationOdia:"ଗତିଶୀଳ ବସ୍ତୁର ଶକ୍ତିକୁ kinetic energy କୁହାଯାଏ।",reviewStatus:"teacher-approved"},
 {id:"S9-ENV-001",class:9,subject:"science",topic:"Natural Resources",subtopic:"ecosystems",difficulty:"core",prompt:"Which gas is required by green plants for photosynthesis?",options:["Oxygen","Nitrogen","Carbon dioxide","Hydrogen"],answerIndex:2,explanationOdia:"ସବୁଜ ଉଦ୍ଭିଦ photosynthesis ସମୟରେ carbon dioxide ବ୍ୟବହାର କରନ୍ତି।",reviewStatus:"teacher-approved"}
];

export const QUESTIONS = [...M8, ...S8, ...M9, ...S9];

export const LIKERT = [
 "Pictures help me understand new ideas.","I learn well when someone explains aloud.","I remember what I write down.",
 "I like learning by doing.","I prefer examples before rules.","I can explain a lesson after hearing it.",
 "I like reading short explanations.","I learn from practice questions.","I like diagrams and charts.",
 "Talking through a problem helps me.","Writing steps helps me remember.","Hands-on activities help me learn."
].map((text,i)=>({id:`L-${String(i+1).padStart(2,"0")}`,text}));
