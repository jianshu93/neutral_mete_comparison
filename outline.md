## Outline for the Neutral-METE Model Comparison Project
### Introductory paragraph

- Many ecological patterns take similar forms across a diverse array of ecosystems and taxonomic groups.
- There are two general explanations for why this might occur.
    1. The patterns represent emergent statistical phenomena, where a broad array of specific processes are expected to give rise to the same pattern.
    2. Across the diverse array of ecological systems these patterns are governed by similar fundamental processes and therefore take similar forms.
- One of the major challenges in evaluating the relative importance of these alternative explanations is that it is often impossible to distinguish between models making similar (or even identical) predictions the a pattern.
- We take advantage of the fact that ecological theories increasingly make predictions for multiple patterns simultaneously to conduct a strong test of the relative fits of a model based on biological processes and one based on emergent patterns.
- By comparing the models using a joint distribution that captures the information for a suite of four patterns, we are able to conduct this strong test using a single direct comparison incorporating information across patterns.


### Briefly recap the concept of the two models
 - Among the existing unified theories of ecology, neutral theory and METE are two most comprehensive ones, which attempt to capture patterns of biodiversity as well as patterns of biomass / energy flux. 
 - While the two take the same quantities as inputs to make predictions for the same set of patterns, they represent two different views on the mechanism behind macroecological patterns.
	 - Neutral theory (*we really need a concise and fancy name for the neutral size with size structure... SNBT maybe?) extends Hubbell's neutral theory of biodiversity to incorporate size structure.
The patterns arise from interactions of biological processes of birth, death, and growth.
	 - In METE, the patterns arise as emergent statistical properties, which represent our best (least-biased) guess of the system given the set of constraints.
 - Previous evaluations of METE have yielded mixed results, while the neutral theory has never been confronted with empirical data.
 - Here we provide the first empirical evaluation of the neutral theory, and by comparing the performance of the two models on the same set of data, attempt to answer the question if the process-based or the constraint-based approach is more appropriate in characterizing community structure in their current state.

### Results and interpretations
 - We compiled data for 60 forest plots that are over 1-ha in size, with X species and Y individuals in total.
 - Performance of the neutral theory and METE was evaluated as their ability to simultaneously capture the distribution of abundance and biomass among species, as well as their potential interactions.
 - This was measured as the likelihood of the models for P(N, M), the joint probability that a species has abundance N and biomass M.
 - The neutral theory had higher likelihood than METE for P(N, M) in all but one community.
 - A closer examination shows that the two models made the same prediction for the SAD, and slightly different but equally good predictions for the size spectrum (*this statement is yet to be validated*). 
The discrepancy lies in their predictions for assigning individuals of different sizes to species.
	 - METE predicts a strong negative correlation between species abundance and average body size, which leads to deviations from the predictions
	 - In contrast, neutral theory predicts that individuals in each species are random samples from the size spectrum, making average body size independent from species abundance.
	 - This prediction is in better agreement with empirical patterns in forest communities, though in almost all communities there are a minority of species that deviate from the expectation of independence.
 - Rooted in the Maximum Entropy Principle (EM), METE yields the least biased prediction for the patterns given the input constraints and its specific configuration.
 - While other configurations with the same set of constraints exist under the EM framework, none of them provide better fits to empirical data than  METE.
 - The fact that the neutral theory outcompetes METE (and the other EM models as far as we know) suggests that the biological processes in the neutral theory indeed provide meaningful information that cannot be fully summarized by the current set of constraints.
 - While the two models have made an ambitious attempt to bridge the gap between patterns of biodiversity and patterns of energy use, they are both in their primitive stage with high potential for future development.
	 - Additional/different constraints can be incorporated into METE. 
	 - The simplified assumption of full neutrality among individuals in the neutral model can be modified to more accurately reflect the relationship between demographic parameters and body size.
 - The constraint-based and the process-based approaches do not necessarily have to be mutually exclusive. Future work needs to be done to elucidate their roles in shaping the structure of ecological systems.