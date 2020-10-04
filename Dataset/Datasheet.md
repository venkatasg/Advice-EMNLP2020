# Datasheet: Advice

Author: Venkata S Govindarajan, Benjamin T. Chen

Organization: University of Texas at Austin


## Motivation

*The questions in this section are primarily intended to encourage
dataset creators to clearly articulate their reasons for creating the
dataset and to promote transparency about funding interests.*

1. **For what purpose was the dataset created?**
Was there a specific task in mind? Was there a specific gap that needed
to be filled? Please provide a description.

    This dataset aims to improve automatic identification of explicit and
    implicit advice in text. Existing methods of advice/suggestion
    identification do not take into account differences in discourse,
    audience, and intra-text variability, which we try to address.

2. **Who created this dataset (e.g. which team, research group) and on
    behalf of which entity (e.g. company, institution, organization)**?

    The raw data was scraped from Reddit, and annotations for the data
    were collection on Amazon Mechanical Turk by Benjamin T Chen, Venkata
    S Govindarajan and Rebecca Warholic under Prof. Katrin Erk and Prof.
    Jessy Li at the University of Texas at Austin.

3. **What support was needed to make this dataset?**

    The costs for annotation were supported by research grants to Prof.
    Jessy Li and Prof. Katrin Erk.

<!--
4. **Any other comments?**

	*Your Answer Here*
 -->


## Composition

*Dataset creators should read through the questions in this section
prior to any data collection and then provide answers once collection
is complete. Most of these questions are intended to provide dataset
consumers with the information they need to make informed decisions
about using the dataset for specific tasks. The answers to some of
these questions reveal information about compliance with the EU’s
General Data Protection Regulation (GDPR) or comparable regulations in
other jurisdictions.*

1. **What do the instances that comprise the dataset represent
(e.g. documents, photos, people, countries)?**

    The dataset comprises sentences from online text.

2. **How many instances are there in total (of each type, if appropriate)?**

    The dataset comprises text from 2 sources &mdash; r/AskParents and
    r/needadvice. 10,594 sentences are from AskParents, and 7862 are from
    needadvice.

3. **Does the dataset contain all possible instances or is it a sample
(not necessarily random) of instances from a larger set?**

    The dataset released was preprocessed during scraping and annotation time as
    described in our paper. Otherwise, this release contains all the instances
    that were used for analysis and modelling in our paper.

4. **What data does each instance consist of?**

    Raw text that has only been tokenized using SpaCy.

5. **Is there a label or target associated with each instance?**

    Each sentence has an associated ID(postID-reply number-sentence number), a
    majority label and a Dawid-Skeene label associated with it. A label value
    of 1 indicates that the sentence is deemed to be advice.

6. **Is any information missing from individual instances?**

    None to the best of our knowledge.

7. **Are relationships between individual instances made explicit (e.g. users' movie ratings, social network links)?**

	Yes, the ID column indicates the relationship between individual instances/
	sentences. The ID column has 3 values separated by hyphens. First is the
	*postID*, which indicates the advice-seeking post for which the sentence
	was a reply to. The second item is the *reply number* as determined by
	Reddit's comment ranking algorithm, and the third item is the *sentence number*,
	which indicates order of sentences within a reply.

8. **Are there recommended data splits (e.g. training, development/validation,
    testing)?**

	We used a train/development/test split of 80-10-10 on *posts* rather than
	sentences so as to retain context for sentences in the same post.

9. **Are there any errors, sources of noise, or redundancies in the dataset?**

	As we explain in Section 3.3 of our paper, we discovered after annotation
	that 69 out of 407 posts in r/AskParents and 98 of 277 posts in r/needadvice
	were annotation with missing post bodies (due to deletion or removal).
	While this might might be a source of annotator uncertainty during advice
	identification, we found that post titles are sufficiently informative for
	succesful annotation.

10. **Is the dataset self-contained, or does it link to or otherwise rely on
    external resources (e.g. websites, tweets, other datasets)?**

	It is self contained as far as only looking at the sentences and replies to
	a post. The context (advice-seeking post) isn't available in this dataset,
	but may be made available separately.

11. **Does the dataset contain data that might be considered confidential
    (e.g. data that is protected by legal privilege or by doctor-patient
    confidentiality, data that includes the content of individuals' non-public
    communications)?**

	No, the data does not contain confidential information. It contains only
	publicly available posts and replies from Reddit, and the dataset we provide
	does not contain any user-identifying information.

12. **Does the dataset contain data that, if viewed directly, might be offensive,
    insulting, threatening, or might otherwise cause anxiety?**

	Yes, our dataset was constructed to investigate the different strategies
	people use to ask for and give advice online. People dp ask for advice on
	sensitive topics online, and our dataset relects this. It contains replies to
	posts seeking advice on various topics including parenting, pregnancy,
	relationships, mental health, and more.

13. **Does the dataset relate to people?**

	No.

14. **Does the dataset identify any subpopulations (e.g. by age, gender)?**

	No.

15. **Is it possible to identify individuals (i.e., one or more natural persons),
    either directly or indirectly (i.e., in combination with other data) from
    the dataset?**

	No. One can only infer that two sentences are from the same person if they
	share a *postID* and *reply number*. However, one can't identify

16. **Does the dataset contain data that might be considered sensitive in any
    way (e.g. data that reveals racial or ethnic origins, sexual orientations,
    religious beliefs, political opinions or union memberships, or locations;
    financial or health data; biometric or genetic data; forms of government
    identification, such as social security numbers; criminal history)?**

	No.

17. **Any other comments?**

	r/AskParents is a community centered around asking for advice related to
	parenting issues. So while, there isn't any personally identifying information
	in our dataset, dataset users need to keep in mind that the majority of
	text from the subreddit is more likely to be from women, since women bear the
	brunt of parenting responsibilities in society.


## Collection

*As with the previous section, dataset creators should read through these
questions prior to any data collection to flag potential issues and then
provide answers once collection is complete. In addition to the goals of the
prior section, the answers to questions here may provide information that allow
others to reconstruct the dataset without access to it.*

1. **How was the data associated with each instance acquired?** Was the data
directly observable (e.g. raw text, movie ratings), reported by subjects (e.g.
survey responses), or indirectly inferred/derived from other data (e.g.
part-of-speech tags, model-based guesses for age or language)? If data was
reported by subjects or indirectly inferred/derived from other data, was the
data validated/verified? If so, please describe how.

    After the raw data was scraped from Reddit (using ???), we collected
    annotations on what spans of text within a reply to an advice-seeking post
    were advice or not using Amazon Mechanical Turk.

2. **What mechanisms or procedures were used to collect the data (e.g. hardware
    apparatus or sensor, manual human curation, software program, software API)?**

	We used the online Amazon Mechinal Turk platform, along with the software
	tools [boto](http://boto.cloudhackers.com/en/latest/) and
	[BRAT](https://www.aclweb.org/anthology/E12-2021/) to collect annotations
	online.

3. **If the dataset is a sample from a larger set, what was the sampling
    strategy (e.g. deterministic, probabilistic with specific sampling
    probabilities)?**

	No sampling was employed.

4. **Who was involved in the data collection process (e.g. students,
    crowdworkers, contractors) and how were they compensated (e.g. how much were
    crowdworkers paid)?**

	Crowdworkers were recruited on Amazon Mechanical Turk from the USA.
	Annotators were compensated $ 0.15 per HIT that they completed. Each HIT
	comprised no more than 5 top-level comments to an advice-seeking post, and
	comment trees were restricted to a depth of 2. Based on pilot studies
	performed by the authors, we estimate that a HIT would take around 5-10
	minutes.

5. **Over what timeframe was the data collected?** Does this timeframe match the creation timeframe of the data associated with the instances (e.g. recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created. Finally, list when the dataset was first published.

	The data was collected between Summer 2019 and February 2020.

7. **Were any ethical review processes conducted (e.g. by an institutional
    review board)?**

	No.

8. **Does the dataset relate to people?** If not, you may skip the remainder of the questions in this section.

	No.

9. **Did you collect the data from the individuals in question directly, or
    obtain it via third parties or other sources (e.g. websites)?**

	The raw data for annotation was scraped online using Reddit APIs. The advice
	labels for spans of text were collected on MTurk by asking annotators to
	perform the task of identifying advice within text.

10. **Were the individuals in question notified about the data collection?**

	The users of the subreddits r/AskParents and r/needadvice were not notified
	of the data collection. The terms of usage on Reddit make user content
	publicly available on the internet, and for use in research materials.
	However, we understand that users may not be comfortable or aware of this,
	which is why we try our best to collect information and preprocess it
	without any identifying information.

	Additionally, we acknowledge the implications of using data available on
	public forums for research (Zimmer, 2018, Ayers et al., 2018) and urge
	researchers and practitioners to respect the privacy of the authors of posts
	in our dataset.

11. **Did the individuals in question consent to the collection and use of their
    data?**

	No (refer to Q10).

12. **If consent was obtained, were the consenting individuals provided with a
    mechanism to revoke their consent in the future or for certain uses?**

	N/A. However, if anyone feels that they would prefer that their content
	from Reddit not be made publicly available on our repository, please do
	[email us](mailto: gvenkata1994@gmail.com), and we will remove your data
	from our dataset.

13. **Has an analysis of the potential impact of the dataset and its use on data
    subjects (e.g. a data protection impact analysis) been conducted?**

	No.


## Preprocessing / Cleaning / Labeling

*Dataset creators should read through these questions prior to any pre-processing, cleaning, or labeling and then provide answers once these tasks are complete. The questions in this section are intended to provide dataset consumers with the information they need to determine whether the “raw” data has been processed in ways that are compatible with their chosen tasks. For example, text that has been converted into a “bag-of-words” is not suitable for tasks involving word order.*

1. **Was any preprocessing/cleaning/labeling of the data done (e.g. discretization or
    bucketing, tokenization, part-of-speech tagging, SIFT feature extraction,
    removal of instances, processing of missing values)?**

	Yes &mdash; all preprocessing steps are explained in Section 3.2 of our
	paper.

2. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled
    data (e.g. to support unanticipated future uses)?**

    The raw dataset has been saved, however it has not been made publicly
    available yet. We might choose to release it in the future, if we don't
    think it would be harmful to do so.

3. **Is the software used to preprocess/clean/label the instances available?**

	Yes, all code used for preprocessing can be found on this repository.


## Uses

*These questions are intended to encourage dataset creators to reflect on the
tasks  for  which  the  dataset  should  and  should  not  be  used.  By
explicitly highlighting these tasks, dataset creators can help dataset consumers
to make informed decisions, thereby avoiding potential risks or harms.*

1. **Has the dataset been used for any tasks already?**

	Yes, it has been used for preliminary modeling experiments in [our paper]()

2. **Is there a repository that links to any or all papers or systems that use
    the dataset?**

	None as of now.

3. **What (other) tasks could the dataset be used for?**

	The data could be used to train a model for generating advice.

4. **Is there anything about the composition of the dataset or the way it was
    collected and preprocessed/cleaned/labeled that might impact future uses?**

	r/AskParents is a community centered around asking for advice related to
	parenting issues. While there isn't any personally identifying information
	in our dataset, dataset users need to keep in mind that the majority of
	text from the subreddit is more likely to be from women, since women bear the
	brunt of parenting responsibilities in society.

5. **Are there tasks for which the dataset should not be used?**

	N/A.


## Distribution

*Dataset creators should provide answers to these questions prior to distributing the dataset either internally within the entity on behalf of which the dataset was created or externally to third parties.*

1. **Will the dataset be distributed to third parties outside of the entity
    (e.g. company, institution, organization) on behalf of which the dataset was
    created?**

	The dataset is made publicly available under the MIT License.

2. **How will the dataset will be distributed (e.g. tarball on website, API,
    GitHub)?**

	The dataset is available on this GitHub repo.

3. **When will the dataset be distributed?**

	Soon.

4. **Will the dataset be distributed under a copyright or other intellectual
    property (IP) license, and/or under applicable terms of use (ToU)?**

	The dataset is publicly available under the MIT License.

5. **Have any third parties imposed IP-based or other restrictions on the data
    associated with the instances?**

	No.

6. **Do any export controls or other regulatory restrictions apply to the
    dataset or to individual instances?**

	No.


## Maintenance

*As with the previous section, dataset creators should provide answers to these questions prior to distributing the dataset. These questions are intended to encourage dataset creators to plan for dataset maintenance and communicate this plan with dataset consumers.*

1. **Who is supporting/hosting/maintaining the dataset?**

	The primary author of the dataset &mdash; Venkata S Govindarajan will be
	maintaining the dataset.

2. **How can the owner/curator/manager of the dataset be contacted (e.g. email address)?**

	The primary author can be contacted over email &mdash;
	[gvenkata1994@gmail.com](mailto: gvenkata1994@gmail.com)

3. **Is there an erratum?** If so, please provide a link or other access point.

	None as of August 2020.

4. **Will the dataset be updated (e.g. to correct labeling errors, add new
    instances, delete instances)?**

	We will look into whether we can upload the data corresponding to the raw
	text of the advice-seeking questions (context).

5. **If the dataset relates to people, are there applicable limits on the
    retention of the data associated with the instances (e.g. were individuals
    in question told that their data would be retained for a fixed period of
    time and then deleted)?**

	N/A.

6. **Will older versions of the dataset continue to be supported/hosted/maintained?**

	Yes, all dataset versions will be available on this GitHub repository.

7. **If others want to extend/augment/build on/contribute to the dataset, is
    there a mechanism for them to do so?**

	We provide the dataset (and all code and models) under the MIT License.

8. **Any other comments?**

	If you have any questions or concerns with the dataset, please don't
	hesitate to [email](mailto: gvenkata1994@gmail.com) me.