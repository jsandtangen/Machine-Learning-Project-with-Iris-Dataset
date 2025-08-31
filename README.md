# Machine Learning Project with Iris Dataset

## Oversikt
Dette prosjektet demonstrerer bruk av ulike maskinlæringsmodeller for klassifisering av Iris-blomsterdata. Vi bruker K-nearest neighbors (KNN), beslutningstre (Decision Tree) og Random Forest-algoritmer for å forutsi arten av Iris-blomster basert på deres egenskaper. Datavisualisering og hyperparameter-tuning av KNN-modellen er også inkludert.

## Innhold
- Importere nødvendige biblioteker
- Laste ned og vise Iris-datasettet
- Dataforbehandling og standardisering
- Sammenligning av forskjellige maskinlæringsmodeller
- Hyperparameter tuning av KNN-modellen
- Evaluering av modeller med forvirringsmatrise og klassifikasjonsrapport
- Visualisering av resultatene

## Forutsetninger
Før du kjører koden, må følgende biblioteker installeres:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Du kan installere disse pakkene ved å bruke pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Kjøring av koden
For å kjøre koden, kan du bruke en Python IDE eller Jupyter Notebook. Sørg for å kopiere koden inn i en Python-fil (.py) eller en Jupyter Notebook (.ipynb).

## Funksjoner
1. **Datavisualisering**:
   - Parplot av Iris-datasettet for å se forholdet mellom egenskapene.
  
2. **Dataforbehandling**:
   - Standardisering av dataene for å forbedre modellens ytelse.

3. **Modellsammenligning**:
   - Bruke kryssvalidering for å evaluere KNN, beslutningstre og Random Forest-modeller.
  
4. **Hyperparameter-tuning**:
   - Grid Search for å finne de beste hyperparametrene for KNN-modellen.

5. **Resultatevaluering**:
   - Forvirringsmatrise og klassifikasjonsrapport for å evaluere modellene.

## Resultater
Etter å ha kjørt koden, vil du se resultater for hver modell, inkludert forvirringsmatriser og klassifikasjonsrapporter. Koden vil også gi deg de beste hyperparametrene for KNN-modellen.

## Lisens
Dette prosjektet er lisensiert under MIT-lisensen. Se LICENSE-filen for detaljer.
