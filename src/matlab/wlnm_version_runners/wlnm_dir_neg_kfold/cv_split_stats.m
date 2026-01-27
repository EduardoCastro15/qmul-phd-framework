function st = cv_split_stats(net, train, test, backbone_mask)
%CV_SPLIT_STATS Mirror your DivideNet_dir_neg split_stats for CV folds.

    m = nnz(net);

    st = struct();
    st.TotalLinks = m;
    st.TrainLinks = nnz(train);
    st.TestLinks  = nnz(test);

    if nargin >= 4 && ~isempty(backbone_mask)
        B = logical(sparse(backbone_mask));
        B = B & (net > 0);
        NB = (net > 0) & ~B;

        st.BackboneTotal        = nnz(B);
        st.NonBackboneTotal     = nnz(NB);
        st.BackboneTrainLinks   = nnz(train & B);
        st.NonBackboneTrainLinks= nnz(train & NB);
        st.BackboneTestLinks    = nnz(test & B);
        st.NonBackboneTestLinks = nnz(test & NB);
    else
        st.BackboneTotal = 0;
        st.NonBackboneTotal = m;
        st.BackboneTrainLinks = 0;
        st.NonBackboneTrainLinks = st.TrainLinks;
        st.BackboneTestLinks = 0;
        st.NonBackboneTestLinks = st.TestLinks;
    end
end
