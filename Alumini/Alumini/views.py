from django.http import HttpResponse
from django.shortcuts import render
import pickle
import easyocr
import cv2
import numpy
import json
import numpy as np
import pandas as pd
from . import matching
from .matching import matching_fn
from .profiling import profile_fn
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

dc=pd.read_csv(r"C:\Users\user\Documents\Django\Alumini\Alumini\Alumini\Companies.csv",sep=';',on_bad_lines='skip',encoding='utf_8_sig')

pkl_file = open(r'C:\Users\user\Documents\Django\Alumini\Alumini\Alumini\profiling.pkl', 'rb')
model = pickle.load(pkl_file)
pkl_file.close()

reader = easyocr.Reader(['en'])
print(type(reader))

def replace_fn(x) :
  if(type(x) is not str):
    return x
  e = ['Ã©' , 'Ã¨' , 'Ã']
  for el in e  :
    x = x.replace(el,'é')
  x = x.replace('âÃ´','o')
  x = x.replace('Ã´','o')
  x = x.replace('Ã§','c')
  x = x.replace('â',"'")
  x = x.replace("Ã","î")
  return x
dc = dc.applymap(replace_fn)


def dashboard (request):
  return render(request,"dashboard.html")



def home (request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        image = cv2.imdecode(numpy.frombuffer(upload.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        bounds = reader.readtext(np.array(image), min_size=0, slope_ths=0.2, ycenter_ths=0.7, height_ths=0.6,
                                 width_ths=0.8, decoder='beamsearch', beamWidth=10)
        
        text = ''
        for i in range(len(bounds)):
            text = text + bounds[i][1] + '\n'

        tokenized_doc = []
        tokenized_doc.append(word_tokenize(text.lower()))

        org_list = []
        df = matching_fn(tokenized_doc)
        organization_list = np.array(df[['organization_url_1','organization_url_2','organization_url_3','organization_url_4','organization_url_5']]).flatten().tolist()
        organization_list.remove("None")
        organization_list = list(map(lambda x : x+ r"about/",organization_list))
        orgainzation_set = set(organization_list)
        ndc = dc.loc[dc["Url"].isin(organization_list)]
        ndc = ndc.fillna('Inconnu')
        json_records = ndc.reset_index().to_json(orient ='records')
        arr = []
        arr = json.loads(json_records)

        contextt = {'d': arr}

        print(contextt)
        return render(request,"Companies.html",contextt)
    return render(request,"Accueil.html")


def companies (request):
    return render(request,"Page-1.html")


def profiles (request):
  sep_count = []
  profiles = {
  'DS':{
    'name':'Data Science',
    'presentation':'La data science est une science interdisciplinaire s’appuyant sur des méthodes scientifiques, des algorithmes, des processus et autres systèmes afin d’exploiter de grands ensembles de données. Les data scientists associent ainsi plusieurs compétences, notamment des connaissances en informatique, statistiques et commerce pour analyser des données collectées auprès des clients ou bien d’autres sources à l’aide de capteurs, de leurs smartphones, de leurs habitudes de navigation sur internet, etc.',
    'image': 'https://img-0.journaldunet.com/zh9zHLYZr-GqRkfcYnoxnwEK9iY=/1500x/smart/740f46b4391740b88b153b80030efef4/ccmcms-jdn/32017948.jpeg',
  },
  'SAE':{
    'name':'SAE',
    'presentation':'l’ingénieur logiciel reçoit une formation plus étendue sur le cycle de développement d’un logiciel, en particulier sur la validation, la vérification, les exigences et les spécifications de logiciels. Ils développent des compétences communes en architecture de systèmes informatiques et en conception de logiciels.',
    'image':'https://www.simplilearn.com/ice9/free_resources_article_thumb/how_to_become_a_software_engineer.jpg',
  },
  'WIN':{
    'name':'WIN',
    'presentation':"L'ingénieur réseau a en charge l'ensemble des réseaux de télécommunications de l'entreprise, qui couvrent les réseaux locaux et distants : téléphonie, internet, intranet... Il est responsable des bonnes performances du réseau, organise et définit les procédures. Il est capable d'identifier et d'anticiper les besoins en informatique de l’entreprise. Pour cela il travaille avec les gestionnaires de réseau qui l'informent sur les baisses de performances et les dysfonctionnements éventuels. Cet ingénieur a une connaissance pointue des différents matériels existants sur le marché et des dernières avancées technologiques. Il doit être curieux, avoir une grande capacité d adaptation.",
    'image':'https://online.norwich.edu/sites/default/files/styles/resource_standard_hero/public/content/resources/header/what_is_a_network_engineer_-_its_role_in_information_security.jpg?itok=kI_R5GHI',
  },
    'SELAM':{
      'name':'SLEAM',
      'presentation':"L'ingénieur systèmes embarqués s'occupe du processus complet qui permet de concevoir une carte électronique, mais aussi de toute la partie programmation. Il assemble les composants électroniques (microprocesseurs), réalise les schémas, les câblages, assure les tests et le suivi de production.",
      'image':"https://ynov-bordeaux.com/wp-content/uploads/2017/02/shutterstock_500045860.jpg"
    },
    'ARCTIC':{
      'name':'ARCTIC',
      'presentation':"Pour simplifier, le cloud computing est la fourniture de services informatiques (notamment des serveurs, du stockage, des bases de données, la gestion réseau, des logiciels, des outils d’analyse, l’intelligence artificielle) via Internet (le cloud) dans le but d’offrir une innovation plus rapide, des ressources flexibles et des économies d’échelle. En règle générale, vous payez uniquement les services cloud que vous utilisez (réduisant ainsi vos coûts d’exploitation), gérez votre infrastructure plus efficacement et adaptez l’échelle des services en fonction des besoins de votre entreprise.",
      'image':"https://www.roberthalf.com/sites/default/files/2019-05/cloud-engineer.jpg"
  },
    'BI':{
      'name':'Buissness Intelligence',
      'presentation':"La Business Intelligence, ou informatique décisionnelle, désigne l’ensemble des technologies permettant aux entreprises d’analyser les données au profit de leurs prises de décisions. L’analyse de données peut être très utile pour assister les entreprises dans leurs prises de décisions. Pour collecter et analyser les données, il est nécessaire d’utiliser une large variété d’outils et de technologies : c’est la Business Intelligence.",
      'image':"https://fedit-production.s3.amazonaws.com/system/image/name_type/43/compress_actuality_teaser_1542789256.jpg"
    },
    'EE':{
      'name':'Electrical Engineer',
      'presentation':"Dans le cadre de son travail, un ingénieur électromécanique a pour tâche de concevoir, réaliser et analyser tout appareil ou outil faisant intervenir l’électricité, l’électronique de puissance et la mécanique. Il s’agit du mariage idéal entre génie mécanique et électrique!",
      'image':"https://media.istockphoto.com/photos/electrician-working-at-electric-panel-picture-id1165561132?k=20&m=1165561132&s=612x612&w=0&h=b0cFzLEoJWuqxIPrws2eMOS4GwxZFKka5efVf8KbXfk="
    },
    'GC':{
      'name':'Genie Civil',
      'presentation':"Le génie civil représente l'ensemble des techniques de constructions civiles. Les ingénieurs civils ou ingénieurs en génie civil s’occupent de la conception, la réalisation, l’exploitation et la réhabilitation d’ouvrages de construction et d’infrastructures dont ils assurent la gestion afin de répondre aux besoins de la société, tout en assurant la sécurité du public et la protection de l’environnement. Très variées, leurs réalisations se répartissent principalement dans cinq grands domaines d’intervention : structures, géotechnique, hydraulique, transport, et environnement.",
      'image':"https://cdn.futura-sciences.com/buildsv6/images/mediumoriginal/7/7/b/77b46739e1_50148266_ingenieur-genie-civil2.jpg"
    },
    'NIDS':{
      'name':'NIDS',
      'presentation':"La sécurité informatique est un terme générique qui s'applique aux réseaux, à Internet, aux points de terminaison, aux API, au cloud, aux applications, aux conteneurs, etc. Elle consiste à établir un ensemble de stratégies de sécurité qui fonctionnent conjointement pour vous aider à protéger vos données numériques.",
      'image':"https://www.axis.ac.cy/wp-content/uploads/2020/05/career_in_ict.jpg"
    },
    'SIM':{
      'name':'SIM',
      'presentation':"L’informatique mobile, qui concerne tous les objets connectés (smartphones, montres, tablettes…) est en fort développement. Que ce soit pour créer des applications adaptées ou pour gérer le système d’exploitation, le secteur recrute, notamment des ingénieurs en informatique mobile.",
      'image':"https://tbcdn.talentbrew.com/company/35016/v3_0/img/software-engineer-hero.jpg"
    },
    'TWIN':{
      'name':'TWIN',
      'presentation':"Le développement Web désigne de manière générale les tâches associées au développement de sites Web destinés à être hébergés via un intranet ou Internet. Le processus de développement web comprend, entre autres, la conception de sites web, le développement de contenu web, l’élaboration de scripts côté client ou côté serveur et la configuration de la sécurité du réseau. Le développement Web est le codage ou la programmation qui permet de faire fonctionner un site Web, selon les exigences du propriétaire. Il traite principalement de l’aspect non conceptuel de la création de sites Web, qui comprend le codage et l’écriture du balisage.",
      'image':"https://englishtribuneimages.blob.core.windows.net/gallary-content/2020/6/2020_6$largeimg_2106819729.jpg"
    },
  
}
  if request.method == 'POST' and request.FILES.get('uploadp',False):
        upload = request.FILES.get('uploadp',False)
        image = cv2.imdecode(numpy.frombuffer(upload.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        bounds = reader.readtext(np.array(image), min_size=0, slope_ths=0.2, ycenter_ths=0.7, height_ths=0.6,
                                 width_ths=0.8, decoder='beamsearch', beamWidth=10)
        
        text = ''
        for i in range(len(bounds)):
            text = text + bounds[i][1] + '\n'

        tokenized_doc = []
        tokenized_doc.append(word_tokenize(text.lower()))

        org_list = []
        df = profile_fn(tokenized_doc)
        speciality = ''
        speciality_list = np.array(df['speciality']).flatten().tolist()
        sep_count = Counter(speciality_list).most_common(3)
        i=0
        spe="SAE"
        while (i<3):
          print(sep_count[i][0],sep_count[i][1])
          if sep_count[i][0] != 'nan':
            spe = sep_count[i][0] 
            break
          i = i + 1
        return render(request,"Profile.html",{'i':profiles[spe]})
  return render(request,"Accueil.html")

