"""The E02 probabilistic inverse: latent measure, target, SMC engine and their records.

`contracts` is the single definition site for every typed record the stage passes between
its parts. Nothing else in E02 redefines one: a second `ThetaRecord` with a different field
order would make two modules disagree about what a particle is while both type-check.
"""
