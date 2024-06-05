# -*- coding: utf-8 -*-

import py2neo
from py2neo import Graph, Node, Relationship
from py2neo import NodeMatcher, RelationshipMatcher
import csv

import spacy
from spacy import displacy
import glob
from tqdm import tqdm
import re

# 连接数据库，一定要加name指定数据库
graph=Graph('http://localhost:7474')

nlp = spacy.load("zh_core_web_md")

node_matcher = NodeMatcher(graph)
relationship_matcher = RelationshipMatcher(graph)

query=''

doc=nlp