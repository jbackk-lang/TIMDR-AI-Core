# Provenance graphs

JSON is the machine-readable graph. DOT is a portable visualisation source.
The execution graph must remain acyclic: a result may depend on a preregistration,
but no result may modify that preregistration, its data manifest, or its controls.
