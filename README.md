# Appeals-to-Housing-and-Communal-services
Housing and communal services wants to learn how to automatically determine the topic of a citizen's appeal. This will speed up the decision-making process and help people in a shorter time. And it will also help relieve the staff.
List of categories:
1. The content of the Apartment Building (Содержание МКД)
2. Landscaping (Благоустройство)
3. Violation of the rules for the use of common property (Нарушение правил пользования общим имуществом)
4. Central heating (Центральное отопление)
5. Facade (Фасад)
6. Water supply (Водоснабжение)
7. Sanitary condition (Санитарное состояние)
8. Illegal information and (or) advertising design (Незаконная информационная и (или) рекламная конструкция)
9. Violation of the order of use of common property (Нарушение порядка пользования общим имуществом)
10. Illegal sale of goods from commercial equipment (counter, box, from the ground) (Незаконная реализация товаров с торгового оборудования (прилавок, ящик, с земли))
11. Basements (Подвалы)
12. The state of advertising or information structures (Состояние рекламных или информационных конструкций)
13. Water disposal (Водоотведение)
14. Damage or malfunction of elements of street infrastructure (Повреждения или неисправность элементов уличной инфраструктуры)
15. Roof (Кровля)
## Dataset problems
The categories formulated are not mutually exclusive. Often citizens address one problem in one category, but they are sent to another. Similar problems may be in different categories, so the dataset is teeming with errors. It will be hard to train a model that can predict all classes well. In the course of the work, an attempt was made to reassign the categories manually.
A lot of appeals are very similar to each other, the meaning is the same, just the word order differs. This affects the metrics on the test sample, they may be overestimated.
## Evaluation
|   | Model                          | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|---|--------------------------------|----------|-------------------|-----------------|--------------|----------------|
| 1 | Distill BERT                   | 0.923    | 0.782             | 0.846           | 0.782        | 0.793          |
| 2 | Distill BERT (reassigned data) | 0.940    | 0.923             | 0.927           | 0.923        | 0.925          |
| 3 | Xgboost with tf-ifd features   | 0.917    | 0.766             | 0.881           | 0.766        | 0.815          |
