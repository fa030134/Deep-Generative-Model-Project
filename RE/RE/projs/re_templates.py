
# TACRED and TACREV
TACRED_LABELS = [
    "no_relation",
    "org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence",
    "per:title",
]

# NLI LABEL TEMPLATES -> template 1 in paper
TACRED_LABEL_TEMPLATES = {
    "no_relation": ["{subj} and {obj} are not related"],
    "per:alternate_names": ["{subj} is also known as {obj}"],
    "per:date_of_birth": [
        "{subj}\u00e2\u0080\u0099s birthday is on {obj}",
        "{subj} was born in {obj}",
    ],
    "per:age": ["{subj} is {obj} years old"],
    "per:country_of_birth": ["{subj} was born in {obj}"],
    "per:stateorprovince_of_birth": ["{subj} was born in {obj}"],
    "per:city_of_birth": ["{subj} was born in {obj}"],
    "per:origin": ["{obj} is the nationality of {subj}"],
    "per:date_of_death": ["{subj} died in {obj}"],
    "per:country_of_death": ["{subj} died in {obj}"],
    "per:stateorprovince_of_death": ["{subj} died in {obj}"],
    "per:city_of_death": ["{subj} died in {obj}"],
    "per:cause_of_death": ["{obj} is the cause of {subj}’s death"],
    "per:countries_of_residence": [
        "{subj} lives in {obj}",
        "{subj} has a legal order to stay in {obj}",
    ],
    "per:stateorprovinces_of_residence": [
        "{subj} lives in {obj}",
        "{subj} has a legal order to stay in {obj}",
    ],
    "per:cities_of_residence": [
        "{subj} lives in {obj}",
        "{subj} has a legal order to stay in {obj}",
    ],
    "per:schools_attended": [
        "{subj} studied in {obj}",
        "{subj} graduated from {obj}",
    ],
    "per:title": ["{subj} is a {obj}"],
    "per:employee_of": [
        "{subj} is member of {obj}",
        "{subj} is an employee of {obj}",
    ],
    "per:religion": [
        "{subj} belongs to {obj} religion",
        "{obj} is the religion of {subj}",
        "{subj} believe in {obj}",
    ],
    "per:spouse": [
        "{subj} is the spouse of {obj}",
        "{subj} is the wife of {obj}",
        "{subj} is the husband of {obj}",
    ],
    "per:parents": [
        "{obj} is the parent of {subj}",
        "{obj} is the mother of {subj}",
        "{obj} is the father of {subj}",
        "{subj} is the son of {obj}",
        "{subj} is the daughter of {obj}",
    ],
    "per:children": [
        "{subj} is the parent of {obj}",
        "{subj} is the mother of {obj}",
        "{subj} is the father of {obj}",
        "{obj} is the son of {subj}",
        "{obj} is the daughter of {subj}",
    ],
    "per:siblings": [
        "{subj} and {obj} are siblings",
        "{subj} is brother of {obj}",
        "{subj} is sister of {obj}",
    ],
    "per:other_family": [
        "{subj} and {obj} are family",
        "{subj} is a brother in law of {obj}",
        "{subj} is a sister in law of {obj}",
        "{subj} is the cousin of {obj}",
        "{subj} is the uncle of {obj}",
        "{subj} is the aunt of {obj}",
        "{subj} is the grandparent of {obj}",
        "{subj} is the grandmother of {obj}",
        "{subj} is the grandson of {obj}",
        "{subj} is the granddaughter of {obj}",
    ],
    "per:charges": [
        "{subj} was convicted of {obj}",
        "{obj} are the charges of {subj}",
    ],
    "org:alternate_names": ["{subj} is also known as {obj}"],
    "org:political/religious_affiliation": [
        "{subj} has political affiliation with {obj}",
        "{subj} has religious affiliation with {obj}",
    ],
    "org:top_members/employees": [
        "{obj} is a high level member of {subj}",
        "{obj} is chairman of {subj}",
        "{obj} is president of {subj}",
        "{obj} is director of {subj}",
    ],
    "org:number_of_employees/members": [
        "{subj} employs nearly {obj} people",
        "{subj} has about {obj} employees",
    ],
    "org:members": ["{obj} is member of {subj}", "{obj} joined {subj}"],
    "org:member_of": ["{subj} is member of {obj}", "{subj} joined {obj}"],
    "org:subsidiaries": [
        "{obj} is a subsidiary of {subj}",
        "{obj} is a branch of {subj}",
    ],
    "org:parents": [
        "{subj} is a subsidiary of {obj}",
        "{subj} is a branch of {obj}",
    ],
    "org:founded_by": ["{subj} was founded by {obj}", "{obj} founded {subj}"],
    "org:founded": [
        "{subj} was founded in {obj}",
        "{subj} was formed in {obj}",
    ],
    "org:dissolved": [
        "{subj} existed until {obj}",
        "{subj} disbanded in {obj}",
        "{subj} dissolved in {obj}",
    ],
    "org:country_of_headquarters": [
        "{subj} has its headquarters in {obj}",
        "{subj} is located in {obj}",
    ],
    "org:stateorprovince_of_headquarters": [
        "{subj} has its headquarters in {obj}",
        "{subj} is located in {obj}",
    ],
    "org:city_of_headquarters": [
        "{subj} has its headquarters in {obj}",
        "{subj} is located in {obj}",
    ],
    "org:shareholders": ["{obj} holds shares in {subj}"],
    "org:website": [
        "{obj} is the URL of {subj}",
        "{obj} is the website of {subj}",
    ],
}

# NLI LABEL TEMPLATES -> template 2 in paper. also what we use in QA4RE
SURE_TACRED_LABEL_TEMPLATES = {
    "no_relation": ["{subj} has no known relations to {obj}"],
    "per:stateorprovince_of_death": ["{subj} died in the state or province {obj}"],
    "per:title": ["{subj} is a {obj}"],
    "org:member_of": ["{subj} is the member of {obj}"],
    "per:other_family": ["{subj} is the other family member of {obj}"],
    "org:country_of_headquarters": ["{subj} has a headquarter in the country {obj}"],
    "org:parents": ["{subj} has the parent company {obj}"],
    "per:stateorprovince_of_birth": ["{subj} was born in the state or province {obj}"],
    "per:spouse": ["{subj} is the spouse of {obj}"],
    "per:origin": ["{subj} has the nationality {obj}"],
    "per:date_of_birth": ["{subj} has birthday on {obj}"],
    "per:schools_attended": ["{subj} studied in {obj}"],
    "org:members": ["{subj} has the member {obj}"],
    "org:founded": ["{subj} was founded in {obj}"],
    "per:stateorprovinces_of_residence": [
        "{subj} lives in the state or province {obj}"
    ],
    "per:date_of_death": ["{subj} died in the date {obj}"],
    "org:shareholders": ["{subj} has shares hold in {obj}"],
    "org:website": ["{subj} has the website {obj}"],
    "org:subsidiaries": ["{subj} owns {obj}"],
    "per:charges": ["{subj} is convicted of {obj}"],
    "org:dissolved": ["{subj} dissolved in {obj}"],
    "org:stateorprovince_of_headquarters": [
        "{subj} has a headquarter in the state or province {obj}"
    ],
    "per:country_of_birth": ["{subj} was born in the country {obj}"],
    "per:siblings": ["{subj} is the siblings of {obj}"],
    "org:top_members/employees": ["{subj} has the high level member {obj}"],
    "per:cause_of_death": ["{subj} died because of {obj}"],
    "per:alternate_names": ["{subj} has the alternate name {obj}"],
    "org:number_of_employees/members": ["{subj} has the number of employees {obj}"],
    "per:cities_of_residence": ["{subj} lives in the city {obj}"],
    "org:city_of_headquarters": ["{subj} has a headquarter in the city {obj}"],
    "per:children": ["{subj} is the parent of {obj}"],
    "per:employee_of": ["{subj} is the employee of {obj}"],
    "org:political/religious_affiliation": [
        "{subj} has political affiliation with {obj}"
    ],
    "per:parents": ["{subj} has the parent {obj}"],
    "per:city_of_birth": ["{subj} was born in the city {obj}"],
    "per:age": ["{subj} has the age {obj}"],
    "per:countries_of_residence": ["{subj} lives in the country {obj}"],
    "org:alternate_names": ["{subj} is also known as {obj}"],
    "per:religion": ["{subj} has the religion {obj}"],
    "per:city_of_death": ["{subj} died in the city {obj}"],
    "per:country_of_death": ["{subj} died in the country {obj}"],
    "org:founded_by": ["{subj} was founded by {obj}"],
}

TACRED_LABEL_VERBALIZER = {
    k: k for k in TACRED_LABELS
}
# for template robustness discussions. template 3 in paper
SURE_2_TACRED_LABEL_TEMPLATES = {
    "no_relation": ["The relation between {subj} and {obj} is not available"],
    "per:stateorprovince_of_death": ["The relation between {subj} and {obj} is state or province of death"],
    "per:title": ["The relation between {subj} and {obj} is title"],
    "org:member_of": ["The relation between {subj} and {obj} is member of"],
    "per:other_family": ["The relation between {subj} and {obj} is other family"],
    "org:country_of_headquarters": ["The relation between {subj} and {obj} is country of headquarters"],
    "org:parents": ["The relation between {subj} and {obj} is parents of the organization"],
    "per:stateorprovince_of_birth": ["The relation between {subj} and {obj} is state or province of birth"],
    "per:spouse": ["The relation between {subj} and {obj} is spouse"],
    "per:origin": ["The relation between {subj} and {obj} is origin"],
    "per:date_of_birth": ["The relation between {subj} and {obj} is date of birth"],
    "per:schools_attended": ["The relation between {subj} and {obj} is schools attended"],
    "org:members": ["The relation between {subj} and {obj} is members"],
    "org:founded": ["The relation between {subj} and {obj} is founded"],
    "per:stateorprovinces_of_residence": ["The relation between {subj} and {obj} is state or province of residence"],
    "per:date_of_death": ["The relation between {subj} and {obj} is date of death"],
    "org:shareholders": ["The relation between {subj} and {obj} is shareholders"],
    "org:website": ["The relation between {subj} and {obj} is website"],
    "org:subsidiaries": ["The relation between {subj} and {obj} is subsidiaries"],
    "per:charges": ["The relation between {subj} and {obj} is charges"],
    "org:dissolved": ["The relation between {subj} and {obj} is dissolved"],
    "org:stateorprovince_of_headquarters": ["The relation between {subj} and {obj} is state or province of headquarters"],
    "per:country_of_birth": ["The relation between {subj} and {obj} is country of birth"],
    "per:siblings": ["The relation between {subj} and {obj} is siblings"],
    "org:top_members/employees": ["The relation between {subj} and {obj} is top members or employees"],
    "per:cause_of_death": ["The relation between {subj} and {obj} is cause of death"],
    "per:alternate_names": ["The relation between {subj} and {obj} is person alternative names"],
    "org:number_of_employees/members": ["The relation between {subj} and {obj} is number of employees or members"],
    "per:cities_of_residence": ["The relation between {subj} and {obj} is cities of residence"],
    "org:city_of_headquarters": ["The relation between {subj} and {obj} is city of headquarters"],
    "per:children": ["The relation between {subj} and {obj} is children"],
    "per:employee_of": ["The relation between {subj} and {obj} is employee of"],
    "org:political/religious_affiliation": ["The relation between {subj} and {obj} is political and religious affiliation"],
    "per:parents": ["The relation between {subj} and {obj} is parents of the person"],
    "per:city_of_birth": ["The relation between {subj} and {obj} is city of birth"],
    "per:age": ["The relation between {subj} and {obj} is age"],
    "per:countries_of_residence": ["The relation between {subj} and {obj} is countries of residence"],
    "org:alternate_names": ["The relation between {subj} and {obj} is organization alternate names"],
    "per:religion": ["The relation between {subj} and {obj} is religion"],
    "per:city_of_death": ["The relation between {subj} and {obj} is city of death"],
    "per:country_of_death": ["The relation between {subj} and {obj} is country of death"],
    "org:founded_by": ["The relation between {subj} and {obj} is founded by"]
}

# for template robustness discussions. template 4 in paper
SURE_3_TACRED_LABEL_TEMPLATES = {
    "no_relation": ["{subj} no relation {obj}"],
    "per:stateorprovince_of_death": ["{subj} person state or province of death {obj}"],
    "per:title": ["{subj} person title {obj}"],
    "org:member_of": ["{subj} organization member of {obj}"],
    "per:other_family": ["{subj} person other family {obj}"],
    "org:country_of_headquarters": ["{subj} organization country of headquarters {obj}"],
    "org:parents": ["{subj} organization parents {obj}"],
    "per:stateorprovince_of_birth": ["{subj} person state or province of birth {obj}"],
    "per:spouse": ["{subj} person spouse {obj}"],
    "per:origin": ["{subj} person origin {obj}"],
    "per:date_of_birth": ["{subj} person date of birth {obj}"],
    "per:schools_attended": ["{subj} person schools attended {obj}"],
    "org:members": ["{subj} organization members {obj}"],
    "org:founded": ["{subj} organization founded {obj}"],
    "per:stateorprovinces_of_residence": ["{subj} person state or provinces of residence {obj}"],
    "per:date_of_death": ["{subj} person date of death {obj}"],
    "org:shareholders": ["{subj} organization shareholders {obj}"],
    "org:website": ["{subj} organization website {obj}"],
    "org:subsidiaries": ["{subj} organization subsidiaries {obj}"],
    "per:charges": ["{subj} person charges {obj}"],
    "org:dissolved": ["{subj} organization dissolved {obj}"],
    "org:stateorprovince_of_headquarters": ["{subj} organization state or province of headquarters {obj}"],
    "per:country_of_birth": ["{subj} person country of birth {obj}"],
    "per:siblings": ["{subj} person siblings {obj}"],
    "org:top_members/employees": ["{subj} organization top members or employees {obj}"],
    "per:cause_of_death": ["{subj} person cause of death {obj}"],
    "per:alternate_names": ["{subj} person alternate names {obj}"],
    "org:number_of_employees/members": ["{subj} organization number of employees or members {obj}"],
    "per:cities_of_residence": ["{subj} person cities of residence {obj}"],
    "org:city_of_headquarters": ["{subj} organization city of headquarters {obj}"],
    "per:children": ["{subj} person children {obj}"],
    "per:employee_of": ["{subj} person employee of {obj}"],
    "org:political/religious_affiliation": ["{subj} organization political or religious affiliation {obj}"],
    "per:parents": ["{subj} person parents {obj}"],
    "per:city_of_birth": ["{subj} person city of birth {obj}"],
    "per:age": ["{subj} person age {obj}"],
    "per:countries_of_residence": ["{subj} person countries of residence {obj}"],
    "org:alternate_names": ["{subj} organization alternate names {obj}"],
    "per:religion": ["{subj} person religion {obj}"],
    "per:city_of_death": ["{subj} person city of death {obj}"],
    "per:country_of_death": ["{subj} person country of death {obj}"],
    "org:founded_by": ["{subj} organization founded by {obj}"]
}

TACRED_VALID_CONDITIONS = {
    "per:alternate_names": ["PERSON:PERSON", "PERSON:MISC"],
    "per:date_of_birth": ["PERSON:DATE"],
    "per:age": ["PERSON:NUMBER", "PERSON:DURATION"],
    "per:country_of_birth": ["PERSON:COUNTRY"],
    "per:stateorprovince_of_birth": ["PERSON:STATE_OR_PROVINCE"],
    "per:city_of_birth": ["PERSON:CITY"],
    "per:origin": ["PERSON:NATIONALITY", "PERSON:COUNTRY", "PERSON:LOCATION"],
    "per:date_of_death": ["PERSON:DATE"],
    "per:country_of_death": ["PERSON:COUNTRY"],
    "per:stateorprovince_of_death": ["PERSON:STATE_OR_PROVINCE"],
    "per:city_of_death": ["PERSON:CITY", "PERSON:LOCATION"],
    "per:cause_of_death": ["PERSON:CAUSE_OF_DEATH"],
    "per:countries_of_residence": ["PERSON:COUNTRY", "PERSON:NATIONALITY"],
    "per:stateorprovinces_of_residence": ["PERSON:STATE_OR_PROVINCE"],
    "per:cities_of_residence": ["PERSON:CITY", "PERSON:LOCATION"],
    "per:schools_attended": ["PERSON:ORGANIZATION"],
    "per:title": ["PERSON:TITLE"],
    "per:employee_of": ["PERSON:ORGANIZATION"],
    "per:religion": ["PERSON:RELIGION"],
    "per:spouse": ["PERSON:PERSON"],
    "per:parents": ["PERSON:PERSON"],
    "per:children": ["PERSON:PERSON"],
    "per:siblings": ["PERSON:PERSON"],
    "per:other_family": ["PERSON:PERSON"],
    "per:charges": ["PERSON:CRIMINAL_CHARGE"],
    "org:alternate_names": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:MISC"],
    "org:political/religious_affiliation": [
        "ORGANIZATION:RELIGION",
        "ORGANIZATION:IDEOLOGY",
    ],
    "org:top_members/employees": ["ORGANIZATION:PERSON"],
    "org:number_of_employees/members": ["ORGANIZATION:NUMBER"],
    "org:members": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
    "org:member_of": [
        "ORGANIZATION:ORGANIZATION",
        "ORGANIZATION:COUNTRY",
        "ORGANIZATION:LOCATION",
        "ORGANIZATION:STATE_OR_PROVINCE",
    ],
    "org:subsidiaries": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:LOCATION"],
    "org:parents": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
    "org:founded_by": ["ORGANIZATION:PERSON"],
    "org:founded": ["ORGANIZATION:DATE"],
    "org:dissolved": ["ORGANIZATION:DATE"],
    "org:country_of_headquarters": ["ORGANIZATION:COUNTRY"],
    "org:stateorprovince_of_headquarters": ["ORGANIZATION:STATE_OR_PROVINCE"],
    "org:city_of_headquarters": ["ORGANIZATION:CITY", "ORGANIZATION:LOCATION"],
    "org:shareholders": ["ORGANIZATION:PERSON", "ORGANIZATION:ORGANIZATION"],
    "org:website": ["ORGANIZATION:URL"],
}

TACRED_VALID_CONDITIONS_REV = {
    "PERSON:PERSON": [
        "per:alternate_names",
        "per:spouse",
        "per:parents",
        "per:children",
        "per:siblings",
        "per:other_family",
    ],
    "PERSON:MISC": ["per:alternate_names"],
    "PERSON:DATE": ["per:date_of_birth", "per:date_of_death"],
    "PERSON:NUMBER": ["per:age"],
    "PERSON:DURATION": ["per:age"],
    "PERSON:COUNTRY": [
        "per:country_of_birth",
        "per:origin",
        "per:country_of_death",
        "per:countries_of_residence",
    ],
    "PERSON:STATE_OR_PROVINCE": [
        "per:stateorprovince_of_birth",
        "per:stateorprovinces_of_residence",
        "per:stateorprovince_of_death",
    ],
    "PERSON:CITY": [
        "per:city_of_birth",
        "per:city_of_death",
        "per:cities_of_residence",
    ],
    "PERSON:NATIONALITY": ["per:origin", "per:countries_of_residence"],
    "PERSON:LOCATION": [
        "per:origin",
        "per:city_of_death",
        "per:cities_of_residence",
    ],
    # 'PERSON:STATE_OR_PROVINCE': [],
    "PERSON:CAUSE_OF_DEATH": ["per:cause_of_death"],
    "PERSON:ORGANIZATION": ["per:schools_attended", "per:employee_of"],
    "PERSON:TITLE": ["per:title"],
    "PERSON:RELIGION": ["per:religion"],
    "PERSON:CRIMINAL_CHARGE": ["per:charges"],
    "ORGANIZATION:ORGANIZATION": [
        "org:alternate_names",
        "org:members",
        "org:member_of",
        "org:subsidiaries",
        "org:parents",
        "org:shareholders",
    ],
    "ORGANIZATION:MISC": ["org:alternate_names"],
    "ORGANIZATION:RELIGION": ["org:political/religious_affiliation"],
    "ORGANIZATION:IDEOLOGY": ["org:political/religious_affiliation"],
    "ORGANIZATION:PERSON": [
        "org:top_members/employees",
        "org:founded_by",
        "org:shareholders",
    ],
    "ORGANIZATION:NUMBER": ["org:number_of_employees/members"],
    "ORGANIZATION:COUNTRY": [
        "org:members",
        "org:member_of",
        "org:parents",
        "org:country_of_headquarters",
    ],
    "ORGANIZATION:LOCATION": [
        "org:member_of",
        "org:subsidiaries",
        "org:city_of_headquarters",
    ],
    "ORGANIZATION:STATE_OR_PROVINCE": [
        "org:member_of",
        "org:stateorprovince_of_headquarters",
    ],
    "ORGANIZATION:DATE": ["org:founded", "org:dissolved"],
    "ORGANIZATION:CITY": ["org:city_of_headquarters"],
    "ORGANIZATION:URL": ["org:website"],
}
# DuIE
DuIE_LABELS=['无关系','主演',
 '目',
 '身高',
 '出生日期',
 '国籍',
 '连载网站',
 '作者',
 '歌手',
 '海拔',
 '出生地',
 '导演',
 '气候',
 '朝代',
 '妻子',
 '丈夫',
 '民族',
 '毕业院校',
 '编剧',
 '出品公司',
 '父亲',
 '出版社',
 '作词',
 '作曲',
 '母亲',
 '成立日期',
 '字',
 '号',
 '所属专辑',
 '所在城市',
 '总部地点',
 '主持人',
 '上映时间',
 '首都',
 '创始人',
 '祖籍',
 '改编自',
 '制片人',
 '注册资本',
 '人口数量',
 '面积',
 '主角',
 '占地面积',
 '嘉宾',
 '简称',
 '董事长',
 '官方语言',
 '邮政编码',
 '专业代码',
 '修业年限',
]
DuIE_VALID_CONDITIONS = {"主演": ["影视作品:人物"],
"目": ["生物:目"],
"身高": ["人物:数字"],
"出生日期": ["人物:日期"],
"国籍": ["人物:国家"],
"连载网站": ["网络小说:网站"],
"作者": ["图书作品:人物"],
"歌手": ["歌曲:人物"],
"海拔": ["地点:数字"],
"出生地": ["人物:地点"],
"导演": ["影视作品:人物"],
"气候": ["行政区:气候"],
"朝代": ["历史人物:文本"],
"妻子": ["人物:人物"],
"丈夫": ["人物:人物"],
"民族": ["人物:文本"],
"毕业院校": ["人物:学校"],
"编剧": ["影视作品:人物"],
"出品公司": ["影视作品:企业"],
"父亲": ["人物:人物"],
"出版社": ["书籍:出版社"],
"作词": ["歌曲:人物"],
"作曲": ["歌曲:人物"],
"母亲": ["人物:人物"],
"成立日期": ["机构:日期"],
"字": ["历史人物:文本"],
"号": ["历史人物:文本"],
"所属专辑": ["歌曲:音乐专辑"],
"所在城市": ["景点:城市"],
"总部地点": ["企业:地点"],
"主持人": ["电视综艺:人物"],
"上映时间": ["影视作品:日期"],
"首都": ["国家:城市"],
"创始人": ["企业:人物"],
"祖籍": ["人物:地点"],
"改编自": ["影视作品:作品"],
"制片人": ["影视作品:人物"],
"注册资本": ["企业:数字"],
"人口数量": ["行政区:数字"],
"面积": ["行政区:数字"],
"主角": ["网络小说:人物"],
"占地面积": ["机构:数字"],
"嘉宾": ["电视综艺:人物"],
"简称": ["机构:文本"],
"董事长": ["企业:人物"],
"官方语言": ["国家:语言"],
"邮政编码": ["行政区:文本"],
"专业代码": ["学科专业:文本"],
"修业年限": ["学科专业:数字"],}
DuIE_VALID_CONDITIONS_REV = {"影视作品:人物": ["主演", "导演", "编剧", "制片人"],
"生物:目": ["目"],
"人物:数字": ["身高"],
"人物:日期": ["出生日期"],
"人物:国家": ["国籍"],
"网络小说:网站": ["连载网站"],
"图书作品:人物": ["作者"],
"歌曲:人物": ["歌手", "作词", "作曲"],
"地点:数字": ["海拔"],
"人物:地点": ["出生地", "祖籍"],
"行政区:气候": ["气候"],
"历史人物:文本": ["朝代", "字", "号"],
"人物:人物": ["妻子", "丈夫", "父亲", "母亲"],
"人物:文本": ["民族"],
"人物:学校": ["毕业院校"],
"影视作品:企业": ["出品公司"],
"书籍:出版社": ["出版社"],
"企业:日期": ["成立日期"],
"歌曲:音乐专辑": ["所属专辑"],
"景点:城市": ["所在城市"],
"企业:地点": ["总部地点"],
"电视综艺:人物": ["主持人", "嘉宾"],
"影视作品:日期": ["上映时间"],
"国家:城市": ["首都"],
"机构:日期": ["成立日期"],
"企业:人物": ["创始人", "董事长"],
"影视作品:作品": ["改编自"],
"企业:数字": ["注册资本"],
"行政区:数字": ["人口数量", "面积"],
"网络小说:人物": ["主角"],
"机构:数字": ["占地面积"],
"机构:文本": ["简称"],
"国家:语言": ["官方语言"],
"行政区:文本": ["邮政编码"],
"学科专业:文本": ["专业代码"],
"学科专业:数字": ["修业年限"],}

DuIE_LABEL_TEMPLATES={"无关系":["{subj}和{obj}无上述关系"],
"主演": ["{subj}的主演是{obj}"],
"目": ["{subj}在分类学上属于{obj}"],
"身高": ["{subj}的身高是{obj}"],
"出生日期": ["{subj}的出生日期是{obj}"],
"国籍": ["{subj}的国籍是{obj}"],
"连载网站": ["{subj}连载于{obj}"],
"作者": ["{subj}的作者是{obj}"],
"歌手": ["{subj}的歌手是{obj}"],
"海拔": ["{subj}的海拔是{obj}"],
"出生地": ["{subj}的出生地是{obj}"],
"导演": ["{subj}的导演是{obj}"],
"气候": ["{subj}的气候是{obj}"],
"朝代": ["{subj}的朝代是{obj}"],
"妻子": ["{subj}的妻子是{obj}"],
"丈夫": ["{subj}的丈夫是{obj}"],
"民族": ["{subj}的民族是{obj}"],
"毕业院校": ["{subj}的毕业院校是{obj}"],
"编剧": ["{subj}的编剧是{obj}"],
"出品公司": ["{subj}的出品公司是{obj}"],
"父亲": ["{subj}的父亲是{obj}"],
"出版社": ["{subj}的出版社是{obj}"],
"作词": ["{subj}的作词是{obj}"],
"作曲": ["{subj}的作曲是{obj}"],
"母亲": ["{subj}的母亲是{obj}"],
"成立日期": ["{subj}的成立日期是{obj}"],
"字": ["{subj}字{obj}"],
"号": ["{subj}号{obj}"],
"所属专辑": ["{subj}所属的专辑是{obj}"],
"所在城市": ["{subj}的所在城市是{obj}"],
"总部地点": ["{subj}的总部位于{obj}"],
"主持人": ["{subj}的主持人是{obj}"],
"上映时间": ["{subj}的上映时间是{obj}"],
"首都": ["{subj}的首都是{obj}"],
"创始人": ["{subj}的创始人是{obj}"],
"祖籍": ["{subj}的祖籍位于{obj}"],
"改编自": ["{subj}改编自{obj}"],
"制片人": ["{subj}的制片人是{obj}"],
"注册资本": ["{subj}的注册资本为{obj}"],
"人口数量": ["{subj}的人口数量是{obj}"],
"面积": ["{subj}的面积是{obj}"],
"主角": ["{subj}的主角是{obj}"],
"占地面积": ["{subj}的占地面积是{obj}"],
"嘉宾": ["{subj}的嘉宾是{obj}"],
"简称": ["{subj}的简称是{obj}"],
"董事长": ["{subj}的董事长是{obj}"],
"官方语言": ["{subj}的官方语言是{obj}"],
"邮政编码": ["{subj}的邮政编码是{obj}"],
"专业代码": ["{subj}的专业代码是{obj}"],
"修业年限": ["{subj}的修业年限是{obj}"],}

DuIE_LABEL_VERBALIZER = {
    k: k for k in DuIE_LABELS
}
# RETACRED

RETACRED_LABELS = [
    "no_relation",
    "per:identity",
    "per:title",
    "per:employee_of",
    "org:top_members/employees",
    "org:alternate_names",
    "org:country_of_branch",
    "org:city_of_branch",
    "org:members",
    "per:age",
    "per:origin",
    "per:spouse",
    "org:member_of",
    "org:stateorprovince_of_branch",
    "per:date_of_death",
    "per:countries_of_residence",
    "per:children",
    "per:cause_of_death",
    "per:stateorprovinces_of_residence",
    "per:cities_of_residence",
    "per:city_of_death",
    "per:parents",
    "per:siblings",
    "org:political/religious_affiliation",
    "per:charges",
    "org:website",
    "per:schools_attended",
    "org:founded_by",
    "org:shareholders",
    "per:religion",
    "per:other_family",
    "per:city_of_birth",
    "org:founded",
    "per:stateorprovince_of_death",
    "per:date_of_birth",
    "org:number_of_employees/members",
    "per:stateorprovince_of_birth",
    "per:country_of_death",
    "per:country_of_birth",
    "org:dissolved",
]

SURE_RETACRED_LABEL_TEMPLATES = {
    "no_relation": ["{subj} has no known relations to {obj}"],
    "per:religion": ["{subj} has the religion {obj}"],
    "org:country_of_branch": ["{subj} has a branch in the country {obj}"],
    "org:stateorprovince_of_branch": [
        "{subj} has a branch in the state or province {obj}"
    ],
    "org:city_of_branch": ["{subj} has a branch in the city {obj}"],
    "org:shareholders": ["{subj} has shares hold in {obj}"],
    "org:top_members/employees": ["{subj} has the high level member {obj}"],
    "org:members": ["{subj} has the member {obj}"],
    "org:website": ["{subj} has the website {obj}"],
    "per:parents": ["{subj} has the parent {obj}"],
    "org:number_of_employees/members": ["{subj} has the number of employees {obj}"],
    "org:political/religious_affiliation": [
        "{subj} has political affiliation with {obj}"
    ],
    "per:age": ["{subj} has the age {obj}"],
    "per:origin": ["{subj} has the nationality {obj}"],
    "org:alternate_names": ["{subj} is also known as {obj}"],
    "per:other_family": ["{subj} is the other family member of {obj}"],
    "per:identity": [
        "{subj} is the identity/pronoun of {obj}",
        "{subj} and {obj} are the same person",
    ],
    # "per:identity": ["{subj} is the personal pronoun of {obj}", "{obj} is the personal pronoun of {subj}", "{subj} and {obj} are the same person"],
    # "per:identity": ["{subj} is the personal pronoun of {obj}", "{subj} and {obj} are the same person"],
    "per:siblings": ["{subj} is the siblings of {obj}"],
    "org:member_of": ["{subj} is the member of {obj}"],
    "per:children": ["{subj} is the parent of {obj}"],
    "per:employee_of": ["{subj} is the employee of {obj}"],
    "per:spouse": ["{subj} is the spouse of {obj}"],
    "org:dissolved": ["{subj} dissolved in {obj}"],
    "per:schools_attended": ["{subj} studied in {obj}"],
    "per:country_of_death": ["{subj} died in the country {obj}"],
    "per:stateorprovince_of_death": ["{subj} died in the state or province {obj}"],
    "per:city_of_death": ["{subj} died in the city {obj}"],
    "per:date_of_death": ["{subj} died in the date {obj}"],
    "per:cause_of_death": ["{subj} died because of {obj}"],
    "org:founded": ["{subj} was founded in {obj}"],
    "org:founded_by": ["{subj} was founded by {obj}"],
    "per:countries_of_residence": ["{subj} lives in the country {obj}"],
    "per:stateorprovinces_of_residence": [
        "{subj} lives in the state or province {obj}"
    ],
    "per:cities_of_residence": ["{subj} lives in the city {obj}"],
    "per:country_of_birth": ["{subj} was born in the country {obj}"],
    "per:stateorprovince_of_birth": ["{subj} was born in the state or province {obj}"],
    "per:city_of_birth": ["{subj} was born in the city {obj}"],
    "per:date_of_birth": ["{subj} has birthday on {obj}"],
    "per:charges": ["{subj} is convicted of {obj}"],
    "per:title": ["{subj} is a {obj}"],
}

RETACRED_VALID_CONDITIONS = {
    "per:identity": ["PERSON:PERSON"],
    "per:title": ["PERSON:TITLE"],
    "per:employee_of": [
        "PERSON:ORGANIZATION",
        "PERSON:CITY",
        "PERSON:STATE_OR_PROVINCE",
        "PERSON:COUNTRY",
        "PERSON:NATIONALITY",
        "PERSON:LOCATION",
    ],
    "org:top_members/employees": ["ORGANIZATION:PERSON"],
    "org:alternate_names": ["ORGANIZATION:ORGANIZATION"],
    "org:country_of_branch": [
        "ORGANIZATION:COUNTRY",
        "ORGANIZATION:LOCATION",
        "ORGANIZATION:STATE_OR_PROVINCE",
        "ORGANIZATION:CITY",
    ],
    "org:city_of_branch": [
        "ORGANIZATION:CITY",
        "ORGANIZATION:LOCATION",
        "ORGANIZATION:STATE_OR_PROVINCE",
        "ORGANIZATION:COUNTRY",
    ],
    "org:members": [
        "ORGANIZATION:ORGANIZATION",
        "ORGANIZATION:COUNTRY",
        "ORGANIZATION:LOCATION",
        "ORGANIZATION:CITY",
        "ORGANIZATION:STATE_OR_PROVINCE",
    ],
    "per:age": ["PERSON:NUMBER", "PERSON:DURATION", "PERSON:TITLE", "PERSON:DATE"],
    "org:member_of": ["ORGANIZATION:ORGANIZATION"],
    "org:stateorprovince_of_branch": [
        "ORGANIZATION:STATE_OR_PROVINCE",
        "ORGANIZATION:LOCATION",
        "ORGANIZATION:COUNTRY",
        "ORGANIZATION:CITY",
    ],
    "per:origin": [
        "PERSON:NATIONALITY",
        "PERSON:COUNTRY",
        "PERSON:CITY",
        "PERSON:LOCATION",
    ],
    "per:children": ["PERSON:PERSON"],
    "per:spouse": ["PERSON:PERSON"],
    "per:stateorprovinces_of_residence": [
        "PERSON:STATE_OR_PROVINCE",
        "PERSON:LOCATION",
        "PERSON:CITY",
    ],
    "per:siblings": ["PERSON:PERSON"],
    "per:countries_of_residence": [
        "PERSON:COUNTRY",
        "PERSON:NATIONALITY",
        "PERSON:LOCATION",
        "PERSON:CITY",
    ],
    "org:political/religious_affiliation": [
        "ORGANIZATION:IDEOLOGY",
        "ORGANIZATION:RELIGION",
    ],
    "per:cities_of_residence": [
        "PERSON:CITY",
        "PERSON:STATE_OR_PROVINCE",
        "PERSON:LOCATION",
        "PERSON:COUNTRY",
    ],
    "per:parents": ["PERSON:PERSON"],
    "per:date_of_death": ["PERSON:DATE", "PERSON:NUMBER", "PERSON:DURATION"],
    "per:schools_attended": ["PERSON:ORGANIZATION"],
    "per:city_of_death": [
        "PERSON:CITY",
        "PERSON:STATE_OR_PROVINCE",
        "PERSON:LOCATION",
        "PERSON:COUNTRY",
    ],
    "org:website": ["ORGANIZATION:URL", "ORGANIZATION:NUMBER"],
    "per:cause_of_death": [
        "PERSON:CAUSE_OF_DEATH",
        "PERSON:CRIMINAL_CHARGE",
        "PERSON:DATE",
        "PERSON:TITLE",
    ],
    "org:founded_by": ["ORGANIZATION:PERSON"],
    "per:other_family": ["PERSON:PERSON"],
    "org:shareholders": ["ORGANIZATION:PERSON", "ORGANIZATION:ORGANIZATION"],
    "per:city_of_birth": [
        "PERSON:CITY",
        "PERSON:LOCATION",
        "PERSON:NATIONALITY",
        "PERSON:STATE_OR_PROVINCE",
    ],
    "per:charges": ["PERSON:CAUSE_OF_DEATH", "PERSON:CRIMINAL_CHARGE", "PERSON:TITLE"],
    "org:founded": ["ORGANIZATION:DATE"],
    "per:religion": ["PERSON:RELIGION", "PERSON:TITLE"],
    "per:date_of_birth": ["PERSON:DATE", "PERSON:NUMBER", "PERSON:DURATION"],
    "per:stateorprovince_of_death": ["PERSON:STATE_OR_PROVINCE"],
    "org:number_of_employees/members": ["ORGANIZATION:NUMBER"],
    "per:stateorprovince_of_birth": [
        "PERSON:STATE_OR_PROVINCE",
        "PERSON:LOCATION",
        "PERSON:NATIONALITY",
    ],
    "per:country_of_birth": [
        "PERSON:COUNTRY",
        "PERSON:STATE_OR_PROVINCE",
        "PERSON:NATIONALITY",
    ],
    "org:dissolved": ["ORGANIZATION:DATE"],
    "per:country_of_death": ["PERSON:NATIONALITY", "PERSON:COUNTRY"],
}

RETACRED_VALID_CONDITIONS_REV = {
    "ORGANIZATION:PERSON": [
        "org:founded_by",
        "org:top_members/employees",
        "org:shareholders",
    ],
    "PERSON:PERSON": [
        "per:identity",
        "per:children",
        "per:spouse",
        "per:siblings",
        "per:other_family",
        "per:parents",
    ],
    "ORGANIZATION:ORGANIZATION": [
        "org:alternate_names",
        "org:members",
        "org:member_of",
        "org:shareholders",
    ],
    "ORGANIZATION:NUMBER": ["org:number_of_employees/members", "org:website"],
    "ORGANIZATION:DATE": ["org:dissolved", "org:founded"],
    "PERSON:NATIONALITY": [
        "per:origin",
        "per:country_of_death",
        "per:employee_of",
        "per:countries_of_residence",
        "per:city_of_birth",
        "per:country_of_birth",
        "per:stateorprovince_of_birth",
    ],
    "PERSON:TITLE": [
        "per:title",
        "per:religion",
        "per:age",
        "per:cause_of_death",
        "per:charges",
    ],
    "PERSON:DATE": [
        "per:date_of_death",
        "per:date_of_birth",
        "per:age",
        "per:cause_of_death",
    ],
    "PERSON:COUNTRY": [
        "per:countries_of_residence",
        "per:origin",
        "per:employee_of",
        "per:country_of_birth",
        "per:country_of_death",
        "per:cities_of_residence",
        "per:city_of_death",
    ],
    "PERSON:ORGANIZATION": ["per:employee_of", "per:schools_attended"],
    "PERSON:CRIMINAL_CHARGE": ["per:cause_of_death", "per:charges"],
    "ORGANIZATION:CITY": [
        "org:city_of_branch",
        "org:members",
        "org:stateorprovince_of_branch",
        "org:country_of_branch",
    ],
    "PERSON:CITY": [
        "per:employee_of",
        "per:city_of_death",
        "per:cities_of_residence",
        "per:city_of_birth",
        "per:origin",
        "per:stateorprovinces_of_residence",
        "per:countries_of_residence",
    ],
    "PERSON:RELIGION": ["per:religion"],
    "PERSON:NUMBER": ["per:age", "per:date_of_birth", "per:date_of_death"],
    "ORGANIZATION:LOCATION": [
        "org:country_of_branch",
        "org:stateorprovince_of_branch",
        "org:city_of_branch",
        "org:members",
    ],
    "PERSON:DURATION": ["per:age", "per:date_of_birth", "per:date_of_death"],
    "ORGANIZATION:RELIGION": ["org:political/religious_affiliation"],
    "ORGANIZATION:URL": ["org:website"],
    "PERSON:STATE_OR_PROVINCE": [
        "per:stateorprovinces_of_residence",
        "per:employee_of",
        "per:stateorprovince_of_birth",
        "per:stateorprovince_of_death",
        "per:cities_of_residence",
        "per:city_of_death",
        "per:country_of_birth",
        "per:city_of_birth",
    ],
    "ORGANIZATION:COUNTRY": [
        "org:country_of_branch",
        "org:members",
        "org:stateorprovince_of_branch",
        "org:city_of_branch",
    ],
    "ORGANIZATION:STATE_OR_PROVINCE": [
        "org:stateorprovince_of_branch",
        "org:city_of_branch",
        "org:country_of_branch",
        "org:members",
    ],
    "ORGANIZATION:IDEOLOGY": ["org:political/religious_affiliation"],
    "PERSON:LOCATION": [
        "per:stateorprovinces_of_residence",
        "per:employee_of",
        "per:city_of_birth",
        "per:cities_of_residence",
        "per:city_of_death",
        "per:stateorprovince_of_birth",
        "per:countries_of_residence",
        "per:origin",
    ],
    "PERSON:CAUSE_OF_DEATH": ["per:charges", "per:cause_of_death"],
}

RETACRED_LABEL_VERBALIZER = {
    k: k for k in RETACRED_LABELS
}

# SEMEVAL
SEMEVAL_LABELS = [
    "Other",
    "Component-Whole(e1,e2)",
    "Component-Whole(e2,e1)",
    "Instrument-Agency(e1,e2)",
    "Instrument-Agency(e2,e1)",
    "Member-Collection(e1,e2)",
    "Member-Collection(e2,e1)",
    "Cause-Effect(e1,e2)",
    "Cause-Effect(e2,e1)",
    "Entity-Destination(e1,e2)",
    "Entity-Destination(e2,e1)",
    "Content-Container(e1,e2)",
    "Content-Container(e2,e1)",
    "Message-Topic(e1,e2)",
    "Message-Topic(e2,e1)",
    "Product-Producer(e1,e2)",
    "Product-Producer(e2,e1)",
    "Entity-Origin(e1,e2)",
    "Entity-Origin(e2,e1)",
]

SURE_SEMEVAL_LABEL_TEMPLATES = {
    "Other": ["{subj} has no known relations to {obj}"],
    "Component-Whole(e1,e2)": ["{subj} is the component of {obj}"],
    "Component-Whole(e2,e1)": ["{obj} is the component of {subj}"],
    "Instrument-Agency(e1,e2)": ["{subj} is the instrument of {obj}"],
    "Instrument-Agency(e2,e1)": ["{obj} is the instrument of {subj}"],
    "Member-Collection(e1,e2)": ["{subj} is the member of {obj}"],
    "Member-Collection(e2,e1)": ["{obj} is the member of {subj}"],
    "Cause-Effect(e1,e2)": ["{subj} has the effect {obj}"],
    "Cause-Effect(e2,e1)": ["{obj} has the effect {subj}"],
    "Entity-Destination(e1,e2)": ["{obj} is the destination of {subj}"],
    "Entity-Destination(e2,e1)": ["{subj} is the destination of {obj}"],
    "Content-Container(e1,e2)": ["{obj} contains {subj}"],
    "Content-Container(e2,e1)": ["{subj} contains {obj}"],
    "Message-Topic(e1,e2)": ["{obj} is the topic of {subj}"],
    "Message-Topic(e2,e1)": ["{subj} is the topic of {obj}"],
    "Product-Producer(e1,e2)": ["{obj} produces {subj}"],
    "Product-Producer(e2,e1)": ["{subj} produces {obj}"],
    "Entity-Origin(e1,e2)": ["{subj} origins from {obj}"],
    "Entity-Origin(e2,e1)": ["{obj} origins from {subj}"],
}

SEMEVAL_VALID_CONDITIONS = {
    "Component-Whole(e1,e2)": ["MISC:MISC"],
    "Component-Whole(e2,e1)": ["MISC:MISC"],
    "Instrument-Agency(e1,e2)": ["MISC:MISC"],
    "Instrument-Agency(e2,e1)": ["MISC:MISC"],
    "Member-Collection(e1,e2)": ["MISC:MISC"],
    "Member-Collection(e2,e1)": ["MISC:MISC"],
    "Cause-Effect(e1,e2)": ["MISC:MISC"],
    "Cause-Effect(e2,e1)": ["MISC:MISC"],
    "Entity-Destination(e1,e2)": ["MISC:MISC"],
    "Entity-Destination(e2,e1)": ["MISC:MISC"],
    "Content-Container(e1,e2)": ["MISC:MISC"],
    "Content-Container(e2,e1)": ["MISC:MISC"],
    "Message-Topic(e1,e2)": ["MISC:MISC"],
    "Message-Topic(e2,e1)": ["MISC:MISC"],
    "Product-Producer(e1,e2)": ["MISC:MISC"],
    "Product-Producer(e2,e1)": ["MISC:MISC"],
    "Entity-Origin(e1,e2)": ["MISC:MISC"],
    "Entity-Origin(e2,e1)": ["MISC:MISC"],
}

SEMEVAL_VALID_CONDITIONS_REV = {
    "MISC:MISC": [
        "Component-Whole(e1,e2)",
        "Component-Whole(e2,e1)",
        "Instrument-Agency(e1,e2)",
        "Instrument-Agency(e2,e1)",
        "Member-Collection(e1,e2)",
        "Member-Collection(e2,e1)",
        "Cause-Effect(e1,e2)",
        "Cause-Effect(e2,e1)",
        "Entity-Destination(e1,e2)",
        "Entity-Destination(e2,e1)",
        "Content-Container(e1,e2)",
        "Content-Container(e2,e1)",
        "Message-Topic(e1,e2)",
        "Message-Topic(e2,e1)",
        "Product-Producer(e1,e2)",
        "Product-Producer(e2,e1)",
        "Entity-Origin(e1,e2)",
        "Entity-Origin(e2,e1)",
    ]
}

SEMEVAL_LABEL_VERBALIZER = {
    k: k for k in SEMEVAL_LABELS
}
