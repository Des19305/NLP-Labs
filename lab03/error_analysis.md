# Error Analysis

## 1. Similarity đúng / Expected Similarity

### 1.1. doctor – physician

**Observed:**
Cosine similarity = **0.970274**

**Expected:**
Hai từ `doctor` và `physician` có nghĩa gần nhau và có thể được sử dụng thay thế cho nhau trong nhiều ngữ cảnh. Vì vậy, kỳ vọng hai từ có mức độ similarity cao.

**Possible explanation:**
Hai từ có ý nghĩa gần nhau và xuất hiện trong các ngữ cảnh liên quan đến y tế, bệnh nhân và bệnh viện. Vì vậy, các context của hai từ có nhiều điểm tương đồng.

**Evidence from corpus:**

**doctor**

> four additional deputies be employed at the fulton county jail and `` a doctor , medical intern or extern be employed for night and weekend duty at the jail ''.

> dr. clark holds an earned doctor of education degree from the university of oklahoma.

> every person will choose his own doctor and hospital ''.

> the plan does not cover doctor bills.

> if the doctor is conscientious , he wants to study the patient.

**physician**

> while working out in sylvania a swelling developed in the knee and he came here to consult the team physician.

> mr. pezza was taken to a nearby johnston physician , dr. allan a. disimone , who treated him.

> first of all , the admitting physician in the va hospital gets the patient as a new patient.

> as a result , it takes a little longer than it would on the outside where the family physician knows about the patient.

> secondly , the va physician knows that when the patient leaves the hospital , he is no longer going to have a chance to visit his patient.

### 1.2. doctor – nurse

**Observed:**
Cosine similarity = **0.962549**

**Expected:**
Hai từ `doctor` và `nurse` được kỳ vọng có similarity tương đối cao vì cả hai đều liên quan đến lĩnh vực chăm sóc sức khỏe và thường xuất hiện trong các ngữ cảnh liên quan đến bệnh nhân hoặc bệnh viện.

**Possible explanation:**
Hai từ thường xuất hiện trong các ngữ cảnh liên quan đến medical care, hospital và patient. Ngoài ra, corpus còn có trường hợp `doctor` và `nurse` xuất hiện trực tiếp trong cùng một câu.

**Evidence from corpus:**

> community visiting nurse services at home for up to 240 days an illness.

> principal clayton w. pohly said he would allow a further collection between classes today , and revealed that y-teen club past surpluses had been used to provide a private hospital nurse monday for mrs. kowalski.

> to our knowledge no nurse in our agency has been employed because of political affiliation.

> tenderly and rather tediously , the camera rivets on the abrupt , deep love of a pretty nurse and a uniformed teacher , complicated by nothing more than a friend they don't want to hurt.

> as i ministered to his needs , i noticed that his face was radiant in spite of his suffering and i learned that he was trusting not only in the skill of his doctor and nurse but also the lord.

### 1.3. doctor – hospital

**Observed:**
Cosine similarity = **0.977811**

**Expected:**
Hai từ `doctor` và `hospital` được kỳ vọng có similarity cao vì chúng thường xuất hiện trong cùng lĩnh vực và trong các ngữ cảnh liên quan đến medical care

**Possible explanation:**
`doctor` và `hospital` thường xuất hiện trong các context liên quan đến bệnh nhân, điều trị và chi phí y tế. Ngoài ra, hai từ còn xuất hiện trực tiếp trong cùng một câu trong corpus

**Evidence from corpus:**

**doctor**

> four additional deputies be employed at the fulton county jail and `` a doctor , medical intern or extern be employed for night and weekend duty at the jail ''.

> dr. clark holds an earned doctor of education degree from the university of oklahoma.

> every person will choose his own doctor and hospital ''.

> the plan does not cover doctor bills.

> if the doctor is conscientious , he wants to study the patient.

**hospital**

> the jury praised the administration and operation of the atlanta police department , the fulton tax commissioner's office , the bellwood and alpharetta prison farms , grady hospital and the fulton health department.

> the third amended the enabling act for creation of the lamar county hospital district , for which a special constitutional amendment previously was adopted.

> money for its construction will be sought later on but in the meantime the state hospital board can accept gifts and donations of a site.

> president kennedy today proposed a mammoth new medical care program whereby social security taxes on 70 million american workers would be raised to pay the hospital and some other medical bills of 14.2 million americans over 65 who are covered by social security or railroad retirement programs.

> full payment of hospital bills for stays up to 90 days for each illness , except that the patient would pay $10 a day of the cost for the first nine days.

---

## 2. Similarity sai / Unexpected Similarity

### 2.1. doctor – representative

**Observed:**
Cosine similarity = **0.982748**

**Expected:**
Hai từ `doctor` và `representative` không có quan hệ ngữ nghĩa trực tiếp rõ ràng, vì vậy similarity được kỳ vọng không cao bằng các cặp như `doctor – physician` hoặc `doctor – nurse`

**Possible explanation:**

* Word embedding học similarity dựa trên **context**, không trực tiếp dựa trên dictionary definition
* Corpus có thể chứa nhiều văn bản báo chí, trong đó các từ khác lĩnh vực vẫn có thể xuất hiện trong những context có một số từ chung
* Kích thước corpus và lượng dữ liệu huấn luyện có thể chưa đủ lớn để tạo ra representation ổn định
* Có thể có ảnh hưởng của **context window** và các từ xuất hiện thường xuyên trong corpus

**Evidence from corpus:**

`representative` xuất hiện chủ yếu trong các context liên quan đến chính trị:

> calling the democrats the `party that lives , breathes and thinks for the good of the people '' , hughes asked ,` a representative democratic vote in the primary for a springboard toward victory in november ''.

> in an apparent effort to head off such a rival primary slate , mr. wagner talked by telephone yesterday with representative charles a. buckley , the bronx democratic leader , and with joseph t. sharkey , the brooklyn democratic leader.

> the panel's action depends on the return of representative james w. trimble , democrat of arkansas , who has been siding with speaker sam rayburn's forces in the rules committee in moving bills to the floor.

> the last obstacle in mrs. geraghty's globe-girdling trip was smoothed out when a representative of syria called upon her to explain that his brother would meet her at the border of that country.

> asked mrs. grace o. peck , representative from multnomah county , of the commission chairman , joseph e. harvey jr.

### 2.2. doctor – father

**Observed:**
Cosine similarity = **0.982697**

**Expected:**
Hai từ `doctor` và `father` không có quan hệ ngữ nghĩa trực tiếp rõ ràng. Vì vậy, similarity cao như vậy là một kết quả bất ngờ

**Possible explanation:**

* `father` và `doctor` có thể xuất hiện trong những loại văn bản tương tự hoặc những câu có các context khác giống nhau
* Word2Vec học quan hệ từ **distributional context**, vì vậy hai từ không đồng nghĩa vẫn có thể có vector gần nhau
* Corpus và lượng dữ liệu huấn luyện có thể chưa đủ để phân biệt tốt tất cả các quan hệ ngữ nghĩa.
* Context window có thể làm một số từ được xem là tương tự nếu chúng xuất hiện trong các vùng context tương tự

**Evidence from corpus:**

`father` chủ yếu xuất hiện trong các context liên quan đến gia đình và hôn nhân:

> he is married and the father of three children.

> their father is charles b. armour.

> because of the recent death of the bride's father , frederick b. hamm , the marriage of miss terry hamm to john bruce parichy will be a small one at noon tomorrow in st. bernadine's church , forest park.

> dr. w. b. i. martin officiated , and the bride was given in marriage by her father.

> the bride was given in marriage by her father.

### 2.3. doctor – 1959

**Observed:**
Cosine similarity = **0.982179**

**Expected:**
`doctor` không được kỳ vọng có similarity cao với một token biểu thị một năm cụ thể như `1959`, vì hai token không có quan hệ ngữ nghĩa trực tiếp

**Possible explanation:**

* `1959` xuất hiện trong nhiều câu thuộc cùng loại văn bản với các từ khác
* corpus có thể chứa nhiều loại văn bản và các context không phản ánh quan hệ semantic trực tiếp
* tần suất xuất hiện của token có thể ảnh hưởng đến vector
* các từ được xem là context của nhau có thể tạo ra similarity cao dù bản thân chúng không gần nghĩa

**Evidence from corpus:**

`1959` xuất hiện trong nhiều context khác nhau:

> among arrests reported by the federal bureau of investigation in 1959 , about half for burglary and larceny involved persons under 18 years of age.

> the announcement that the city would sue for recovery on the performance bond was made by city solicitor david berger at a press conference following a meeting in the morning with wagner and other officials of the city and the ptc as well as representatives of an engineering firm that was pulled off the el project before its completion in 1959.

> gladden has been an outspoken critic of the present city administration and led his union's battle against the teamsters , which began organizing city firemen in 1959.

> ierulli , 29 , has been practicing in portland since november , 1959.

> but he was scholastically ineligible in 1959 and merely present last season.

---