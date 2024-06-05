from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, CSVLoader

from py2neo import Graph, Node, Relationship
from py2neo import NodeMatcher, RelationshipMatcher
import spacy
from tqdm import tqdm
import re

data_dir = ''

txt_loader = DirectoryLoader(
    path=data_dir,
    glob='**/*.txt',
    show_progress=True,
    use_multithreading=True,
    loader_cls=TextLoader,
    # loader_kwargs=text_loader_kwargs,
)

txt_docs = txt_loader.load()
#print(txt_docs)


graph=Graph('http://localhost:7474')

nlp = spacy.load("zh_core_web_trf")

patterns = r'。|？|！|（|）|；|\r'

for doc in tqdm(txt_docs):
    #print(doc)
    #string_text = [doc[i].page_content for i in range(len(doc))]
    lines = re.split(patterns,doc.page_content)
    for line in tqdm(lines):
        #print(line)
        line_node=Node("TEXT",text=line)
        graph.merge(line_node,"TEXT","text")
        if line!="":
            doc=nlp(line)
            # for token in doc:
            #     #print(token.text,token.pos_,token.tag_)
            #     print(token.text,token.dep_,token.head)
            # for chunk in doc.noun_chunks:
            #     print(chunk.text)
            #print("===========================")
            for ent in doc.ents:
                #print (ent.text, ent.label_)
                ent_node=Node(ent.label_,text=ent.text)
                graph.merge(ent_node,ent.label_,"text")
                relation=Relationship(line_node,"include",ent_node)
                graph.create(relation)



