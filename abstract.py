# coding=utf-8
__author__ = 'rcastro'
import re


title_ptn = re.compile('title: (.+)[\r\n]')
keywords_ptn = re.compile('keywords: (.+):')
abstract_ptn = re.compile('abstract: (.+)[\r\n]')
number_token = re.compile('[\d]+[^\s]*')


class Abstract(object):
    '''
    Reference Type:  Journal Article
    Record Number: 28669
    Author: Zuo, Y. and Zhu, Z.
    Year: 2014
    Title: Simultaneous identification and quantification of 4-cumylphenol, 2,4-bis-(dimethylbenzyl)phenol and bisphenol A in prawn Macrobrachium rosenbergii
    Journal: Chemosphere
    Volume: 107
    Pages: 447-453
    Type of Article: Article
    Short Title: Simultaneous identification and quantification of 4-cumylphenol, 2,4-bis-(dimethylbenzyl)phenol and bisphenol A in prawn Macrobrachium rosenbergii
    DOI: 10.1016/j.chemosphere.2014.01.058
    Keywords: 2,4-Bis-(dimethylbenzyl)phenol
    4-Cumylphenol
    Alkylphenol
    Bisphenol A
    Gas chromatography-mass spectrometry
    Prawn
    Abstract: Bisphenol A (BPA), 4-cumylphenol (4-CP) and 2,4-bis-(dimethylbenzyl)phenol (2,4-DCP) are all high production volume chemicals and widely used in plastic and other consumer products. During the past two decades, BPA has attracted a great deal of scientific and public attention due to its presence in the environment and estrogenic property. Although 4-CP and 2,4-DCP are much more estrogenic and toxic than BPA, little information is available about their occurrence and fate in the environment. In this study, a rapid, selective, accurate and reliable analytical method was developed for the simultaneous determination of 4-CP, 2,4-DCP and BPA in prawn Macrobrachium rosenbergii. The method comprises an ultrasound-accelerated extraction followed by capillary gas chromatographic (GC) separation. The detection limits range from 1.50 to 36.4ngkg-1 for the three alkylphenols. The calibration curves are linear over the concentration range tested with the coefficients of determination, R2, greater than 0.994. The developed method was successfully applied to the simultaneous determination of 4-CP, 2,4-DCP and BPA in prawn samples. The peak identification was confirmed using GC-MS. Bisphenol A, 2,4-bis-(dimethylbenzyl)phenol and 4-cumylphenol were found in prawn samples in the concentration ranges of 0.67-5.51, 0.36-1.61, and 0.00-1.96ngg-1 (wet weight), respectively. All relative standard deviations are less than 4.8%. At these environmentally relevant concentration levels, 4-CP, 2,4-DCP and BPA may affect the reproduction and development of aquatic organisms, including negative influence on crustaceans' larval survival, molting, metamorphosis and shell hardening. This is the first study reported on the occurrence of 4-CP, 2,4-DCP and BPA in prawn M. rosenbergii. © 2014 Elsevier Ltd.
    Notes: Cited By :4
    Export Date: 13 March 2016
    References: Alexander, H.C., Dill, D.C., Smith, L.W., Guiney, P.D., Dorn, P.B., Bisphenol A: acute aquatic toxicity (1988) Environ. Toxicol. Chem., 7, pp. 19-26; Belfroid, A., van Velzen, M., van der Horst, B., Vethaak, D., Occurrence of bisphenol A in surface water and uptake in fish: evaluation of field measurements (2002) Chemosphere, 49, pp. 97-103; Berkner, S., Streck, G., Herrmann, R., Development and validation of a method for determination of trace levels of alkylphenols and bisphenol A in atmospheric samples (2004) Chemosphere, 54, pp. 575-584; Biggers, W.J., Laufer, H., Identification of juvenile hormone-active alkylphenols in the lobster Homarus americanus and in marine sediments (2004) Biol. Bull., 206, pp. 13-24; Calafat, A.M., Ye, X., Wong, L.Y., Reidy, J.A., Needham, L.L., Exposure of the U.S. population to bisphenol A and 4-tertiaty-octylphenol: 2003-2004 (2008) Environ. Health Perspect., 116, pp. 39-44; Chen, M., Jacobs, M., Laufer, H., (2010), Competition of tyrosine with alkylphenols during shell hardening in the new cuticles of lobsters. SICB, Seattle, WA, January 3-7, 2010Dodds, E.C., Lawson, W., Synthetic oestrogenic agents without the phenanthrene nucleus (1936) Nature, 137, pp. 996-997; Fukata, H., Miyagawa, H., Yamazaki, N., Mori, C., Comparison of ELISA- and LC-MS-based methodologies for the exposure assessment of bisphenol A (2006) Toxicol. Mech. Methods, 16, pp. 427-430; Gould, J.C., Leonard, L.S., Maness, S.C., Wagner, B.L., Conner, K., Bisphenol A interacts with the estrogen receptor alpha in a distinct manner from estradiol (1998) Mol. Cell Endocrine, 142, pp. 203-214; Kuiper, G.G., Lemmen, J.G., Carlsson, B., Corton, J.C., Safe, S.H., Interaction of estrogenic chemicals and phytoestrogens with estrogen receptor beta (1998) Endocrinology, 139, pp. 4252-4263; Lahnsteiner, F., Berger, B., Kletz, M., Weismann, T., Effect of bisphenol A on maturation and quality of semen and eggs in the brown trout, Salmo trutta f. fario (2005) Aquat. Toxicol., 75, pp. 213-224; Laufer, H., Chen, M., Baclaski, B., Bobbitt, J.M., Stuart, J.D., Zuo, Y., Jacobs, M.W., Multiple factors in marine environments affecting lobster survival, development, and growth, with emphasis on alkylphenols: a perspective (2013) Can. J. Fish. Aquat. Sci., 70, pp. 1588-1600; Mandich, A., Bottero, S., Benfenati, E., Cevasco, A., Erratico, C., Maggioni, S., Massari, A., Vigano, L., In vivo exposure of carp to graded concentrations of bisphenol A (2007) Gen. Comp. Endocrinol., 153, pp. 15-24; Melnick, R., Lucier, G., Wolfe, W., Hall, R., Stancel, G., Prins, G., Gallo, M., Kohn, M., Summary of the national toxicology program's report of the endocrine disruptors low-dose peer review (2002) Environ. Health Perspect., 110, pp. 427-431; Mita, L., Bianco, M., Viggiano, E., Zollo, F., Bencivenga, U., Sica, V., Monaco, G., Mita, D.C., Bisphenol A content in fish caught in two different sites of the Tyrrhenian Sea (Italy) (2011) Chemosphere, 82, pp. 405-410; Oehlmann, J., Schulte-Oehlmann, U., Kloas, W., Jagnytsch, O., Lutz, I., Kusk, K.O., Wollenberger, L., Tyler, C.R., A critical analysis of the biological impacts of plasticizers on wildlife (2009) Phil. Trans. R. Soc. B, 364, pp. 2047-2062; Okuda, K., Takiguchi, M., Yoshihara, S., In vivoestrogenic potential of 4-methyl-2,4-bis(4-hydroxyphenyl)pent-1-ene, an active metabolite of bisphenol A, in uterus of ovariectomized rat (2010) Toxicol. Lett., 197, pp. 7-11; Ryan, B.C., Hotchkiss, A.K., Crofton, K.M., Gray, L.E., In utero and lactational exposure to bisphenol A, in contrast to ethinyl estradiol, does not alter sexually dimorphic behavior, puberty, fertility, and anatomy of female LE rats (2010) Toxicol. Sci., 114, pp. 133-148; Shao, B., Han, H., Li, D., Zhao, R., Meng, J., Ma, Y., Analysis of nonylhenol, octylphenol and bisphenol A in animal tissues by liquid chromatography-tandem mass spectrometry with accelerated solvent extraction (2005) Se Pu, 23, pp. 362-365; Shi, T., (2012), Determination of Trace Amounts of Bisphenol A on Banknotes and Receipts. University of Massachusetts Dartmouth, MS ThesisSohoni, P., Tyler, C.R., Hurd, K., Caunter, J., Hetheride, M., Williams, T., Woods, C., Sumpter, J.P., Reproductive effects of longterm exposure to bisphenol A in the fathead minnow (Pimephales promelas) (2001) Environ. Sci. Technol., 35, pp. 2917-2925; Staples, C.A., Dorn, P.B., Klecka, G.M., Block, S.T., Harris, L.R., A review of the environmental fate, effects and exposure of bisphenol A (1998) Chemosphere, 36, pp. 2149-2173; Stuart, J.D., Capulong, C.P., Launer, K.D., Pan, X., Analyses of phenolic endocrine disrupting chemicals in marine samples by both gas and liquid chromatography-mass spectrometry (2005) J. Chromatogr. A, 24, p. 136145; Terasaki, M., Shiraishi, F., Nishikawa, T., Edmonds, J.S., Morita, M., Makino, M., Estrogenic activity of impurities in industrial grade bisphenol A (2005) Environ. Sci. Technol., 39, pp. 3703-3707; (2009), U.S. EPA, 2009. Screening-Level Hazard Characterization, Alkylphenols Category. Hazard Characterization Document, Washington DC, SeptemberVandenberg, L.N., Chahoud, I., Heindel, J.J., Padmanabhan, V., Paumgartten, F.J., Schoenfelder, G., Urinary, circulating, and tissue biomonitoring studies indicate widespread exposure to bisphenol A (2010) Environ. Health Perspect., 118, pp. 1055-1070; Wang, C., Zuo, Y., Ultrasound-assisted hydrolysis and gas chromatography-mass spectrometric determination of phenolic compounds in cranberry products (2011) Food Chem., 128, pp. 562-568; Wang, B., Huang, B., Jin, W., Zhao, S., Li, F., Hu, P., Pan, X., Occurrence, distribution, and sources of six phenolic endocrine disrupting chemicals in the 22 river estuaries around Dianchi Lake in China (2013) Environ. Sci. Pollut. Res., 20, pp. 3185-3194; (2010), WHO, 2011. Toxicological and Health Aspects of Bisphenol A, Report of joint FAO/WHO expert meeting 2-5 November 2010 and Report of Stakeholder Meeting on Bisphenol A, Ottawa, Canada, 1 November(2007), WSP, . Screening of Bisphenol A in Fish from Swedish WatersYoshida, T., Horie, M., Hoshino, Y., Nakazawa, H., Determination of bisphenol A in canned vegetables and fruit by high performance liquid chromatography (2001) Food Addit. Contam., 18, pp. 69-75; Yoshihara, S., Mizutare, T., Makishima, M., Fujimoto, S.N., Potent estrogenic metabolites of bisphenol A and bisphenol B formed by rate liver S9 fraction: their structures and estrogenic potency (2004) Toxicol. Sci., 78, pp. 50-59; Zhu, Z., (2012), Determination of Bisphenol A and Other Alkylphenols in American Lobster Homarus americanus. University of Massachusetts Dartmouth, MS ThesisZhu, Z., Zuo, Y., Bisphenol A and other alkylphenols in the environment - occurrence, fate, health effects and analytical techniques (2013) Adv. Environ. Res., 2 (3), pp. 179-202; Zuo, Y., (2014) High-Performance Liquid Chromatography (HPLC): Principles, Procedures and Practices, , Nova Science Publishers Inc; Zuo, Y., Lin, Y., Solvent effects on the silylation-gas chromatography-mass spectrometric determination of natural and synthetic estrogenic steroid hormones (2007) Chemosphere, 69, pp. 1175-1176; Zuo, Y., Zhang, L., Wu, J., Fritz, J., Medeiros, S., Rego, C., Ultrasonic extraction and capillary gas chromatography determination of nicotine in pharmaceutical formulations (2004) Anal. Chim. Acta, 526, pp. 35-39; Zuo, Y., Zhang, K., Deng, Y., Occurrence and photochemical degradation of 17α-ethinylestradiol in Acushnet River Estuary (2006) Chemosphere, 63, pp. 1583-1590; Zuo, Y., Zhang, K., Wu, J., Men, B., He, M., Determination of o-phthalic acid in snow and its photochemical degradation by capillary gas chromatography coupled with flame ionization and mass spectrometric detection (2011) Chemosphere, 83, pp. 1014-1019; Zuo, Y., Zhang, K., Zhou, S., Determination of estrogenic steroids and microbial and photochemical degradation of 17α-ethinylestradiol (EE2) in lake surface water, a case study (2013) Environ. Sci. Process. Impacts, 15, pp. 1529-1535
    URL: http://www.scopus.com/inward/record.url?eid=2-s2.0-84901634826&partnerID=40&md5=f39eb099b741c0bee98bcce48456d75d
    Author Address: Department of Chemistry and Biochemistry, University of Massachusetts Dartmouth, North Dartmouth, 285 Old Westport Road, North Dartmouth, MA 02747, United States
    University of Massachusetts Graduate School of Marine Sciences and Technology, 285 Old Westport Road, North Dartmouth, MA 02747, United States
    Name of Database: Scopus
    '''
    def __init__(self, abstract_text):
        '''
        Keywords: Monoclonal antibody
        Penaeus
        Proteins
        Shrimp
        Abstract:
        '''
        abstract_text = abstract_text.lower()
        self.title = re.findall(title_ptn, abstract_text)[0].strip()
        ar = abstract_text.split('keywords: ')
        keywords = ''
        if len(ar)>1:
            keywords = abstract_text.split('keywords: ')[1].split(':')[0].split('\r\n')[:-1] #re.findall(keywords_ptn, abstract_text)
        self.keywords = keywords
        self.text = self.title
        try:
            self.text += ' ' + re.findall(abstract_ptn, abstract_text)[0].strip()
        except:
            pass
        if self.text:
            self.text = re.sub(r'[\d]+[^\s]*', ' ', self.text)
        self.references = ''
