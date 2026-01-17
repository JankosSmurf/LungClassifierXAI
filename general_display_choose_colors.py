import matplotlib

matplotlib.use('TkAgg')

from newest_classifier import *
from new_classifier import *
from old_classifier import *
from segmenters import MySegmenter, LungSegmenter
from my_color_palette import *

import shap
from lime import lime_image
from skimage.color import gray2rgb
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

### Different explainers

def lime_4_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet4, 'weights/new_classify2d_4_outs.ckpt', device='cpu')
    classifier_wrapper.fit()
    segmenter = MySegmenter("felzenszwalb", 200, 0.5, 50)
    segmenter_with_mask = LungSegmenter(segmenter)

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        sample_to_explain,
        classifier_wrapper.predict_proba,
        top_labels=4,
        hide_color=[55, 55, 55],
        num_samples=10000,
        segmentation_fn=segmenter_with_mask
    )
    global_max = 0
    for i in range(4):
        mask_vls = []
        for h in range(len(explanation.local_exp[i])):
            mask_vls.append(explanation.local_exp[i][h][1])
            mask_vls.append(-explanation.local_exp[i][h][1])
        loc_max = max(mask_vls)
        global_max = max(global_max, loc_max)
    lime_values_list = []
    global_lime_values_list = []
    segments = explanation.segments - np.min(explanation.segments)
    for i in range(4):
        lime_values = np.zeros([64, 64])
        global_lime_values = np.zeros([64, 64])
        mask_vals = {}
        for h in range(len(explanation.local_exp[i])):
            mask_vals[explanation.local_exp[i][h][0]] = explanation.local_exp[i][h][1]
        sh_min = min(mask_vals.values())
        sh_max = max(mask_vals.values())
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = mask_vals[segments[s, t]] / local_max
                lime_values[s, t] = 0.5 + (normd / 2)
                normg = mask_vals[segments[s, t]] / global_max
                global_lime_values[s, t] = 0.5 + (normg / 2)
        lime_values_list.append(lime_values)
        global_lime_values_list.append(global_lime_values)

    return lime_values_list, global_lime_values_list, greens

def lime_16_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(ClassifyNet16, 'weights/old_classify2d_16_outs.ckpt', device='cpu')
    classifier_wrapper.fit()
    segmenter = MySegmenter("felzenszwalb", 200, 0.5, 50)
    segmenter_with_mask = LungSegmenter(segmenter)

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        sample_to_explain,
        classifier_wrapper.predict_proba,
        top_labels=16,
        hide_color=[55, 55, 55],
        num_samples=10000,
        segmentation_fn=segmenter_with_mask
    )
    global_max = 0
    for i in range(16):
        mask_vls = []
        for h in range(len(explanation.local_exp[i])):
            mask_vls.append(explanation.local_exp[i][h][1])
            mask_vls.append(-explanation.local_exp[i][h][1])
        loc_max = max(mask_vls)
        global_max = max(global_max, loc_max)
    lime_values_list = []
    global_lime_values_list = []
    segments = explanation.segments - np.min(explanation.segments)
    for i in range(16):
        lime_values = np.zeros([64, 64])
        global_lime_values = np.zeros([64, 64])
        mask_vals = {}
        for h in range(len(explanation.local_exp[i])):
            mask_vals[explanation.local_exp[i][h][0]] = explanation.local_exp[i][h][1]
        sh_min = min(mask_vals.values())
        sh_max = max(mask_vals.values())
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = mask_vals[segments[s, t]] / local_max
                lime_values[s, t] = 0.5 + (normd / 2)
                normg = mask_vals[segments[s, t]] / global_max
                global_lime_values[s, t] = 0.5 + (normg / 2)
        lime_values_list.append(lime_values)
        global_lime_values_list.append(global_lime_values)

    return lime_values_list, global_lime_values_list, greens

def lime_5_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet, 'weights/new_classify2d.ckpt', device='cpu')
    classifier_wrapper.fit()
    segmenter = MySegmenter("felzenszwalb", 200, 0.5, 50)
    segmenter_with_mask = LungSegmenter(segmenter)

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        sample_to_explain,
        classifier_wrapper.predict_proba,
        top_labels=5,
        hide_color=[55, 55, 55],
        num_samples=10000,
        segmentation_fn=segmenter_with_mask
    )
    global_max = 0
    for i in range(4):
        mask_vls = []
        for h in range(len(explanation.local_exp[i])):
            mask_vls.append(explanation.local_exp[i][h][1])
            mask_vls.append(-explanation.local_exp[i][h][1])
        loc_max = max(mask_vls)
        global_max = max(global_max, loc_max)
    lime_values_list = []
    global_lime_values_list = []
    segments = explanation.segments - np.min(explanation.segments)
    for i in range(4):
        lime_values = np.zeros([64, 64])
        global_lime_values = np.zeros([64, 64])
        mask_vals = {}
        for h in range(len(explanation.local_exp[i])):
            mask_vals[explanation.local_exp[i][h][0]] = explanation.local_exp[i][h][1]
        sh_min = min(mask_vals.values())
        sh_max = max(mask_vals.values())
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = mask_vals[segments[s, t]] / local_max
                lime_values[s, t] = 0.5 + (normd / 2)
                normg = mask_vals[segments[s, t]] / global_max
                global_lime_values[s, t] = 0.5 + (normg / 2)
        lime_values_list.append(lime_values)
        global_lime_values_list.append(global_lime_values)

    return lime_values_list, global_lime_values_list, greens

def lime_17_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(ClassifyNet, 'weights/classify2d.ckpt', device='cpu')
    classifier_wrapper.fit()
    segmenter = MySegmenter("felzenszwalb", 200, 0.5, 50)
    segmenter_with_mask = LungSegmenter(segmenter)

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        sample_to_explain,
        classifier_wrapper.predict_proba,
        top_labels=17,
        hide_color=[55, 55, 55],
        num_samples=10000,
        segmentation_fn=segmenter_with_mask
    )
    global_max = 0
    for i in range(16):
        mask_vls = []
        for h in range(len(explanation.local_exp[i])):
            mask_vls.append(explanation.local_exp[i][h][1])
            mask_vls.append(-explanation.local_exp[i][h][1])
        loc_max = max(mask_vls)
        global_max = max(global_max, loc_max)
    lime_values_list = []
    global_lime_values_list = []
    segments = explanation.segments - np.min(explanation.segments)
    for i in range(16):
        lime_values = np.zeros([64, 64])
        global_lime_values = np.zeros([64, 64])
        mask_vals = {}
        for h in range(len(explanation.local_exp[i])):
            mask_vals[explanation.local_exp[i][h][0]] = explanation.local_exp[i][h][1]
        sh_min = min(mask_vals.values())
        sh_max = max(mask_vals.values())
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = mask_vals[segments[s, t]] / local_max
                lime_values[s, t] = 0.5 + (normd / 2)
                normg = mask_vals[segments[s, t]] / global_max
                global_lime_values[s, t] = 0.5 + (normg / 2)
        lime_values_list.append(lime_values)
        global_lime_values_list.append(global_lime_values)

    return lime_values_list, global_lime_values_list, greens

def lime_6_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet6, 'weights/newest_classify2d_6_outs.ckpt', device='cpu')
    classifier_wrapper.fit()
    segmenter = MySegmenter("felzenszwalb", 200, 0.5, 50)
    segmenter_with_mask = LungSegmenter(segmenter)

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        sample_to_explain,
        classifier_wrapper.predict_proba,
        top_labels=6,
        hide_color=[55, 55, 55],
        num_samples=10000,
        segmentation_fn=segmenter_with_mask
    )
    global_max = 0
    for i in range(6):
        mask_vls = []
        for h in range(len(explanation.local_exp[i])):
            mask_vls.append(explanation.local_exp[i][h][1])
            mask_vls.append(-explanation.local_exp[i][h][1])
        loc_max = max(mask_vls)
        global_max = max(global_max, loc_max)
    lime_values_list = []
    global_lime_values_list = []
    segments = explanation.segments - np.min(explanation.segments)
    for i in range(6):
        lime_values = np.zeros([64, 64])
        global_lime_values = np.zeros([64, 64])
        mask_vals = {}
        for h in range(len(explanation.local_exp[i])):
            mask_vals[explanation.local_exp[i][h][0]] = explanation.local_exp[i][h][1]
        sh_min = min(mask_vals.values())
        sh_max = max(mask_vals.values())
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = mask_vals[segments[s, t]] / local_max
                lime_values[s, t] = 0.5 + (normd / 2)
                normg = mask_vals[segments[s, t]] / global_max
                global_lime_values[s, t] = 0.5 + (normg / 2)
        lime_values_list.append(lime_values)
        global_lime_values_list.append(global_lime_values)

    return lime_values_list, global_lime_values_list, greens

def lime_6_ones_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet6, 'weights/newest_classify2d_ones_6_outs.ckpt', device='cpu')
    classifier_wrapper.fit()
    segmenter = MySegmenter("felzenszwalb", 200, 0.5, 50)
    segmenter_with_mask = LungSegmenter(segmenter)

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        sample_to_explain,
        classifier_wrapper.predict_proba,
        top_labels=6,
        hide_color=[55, 55, 55],
        num_samples=10000,
        segmentation_fn=segmenter_with_mask
    )
    global_max = 0
    for i in range(6):
        mask_vls = []
        for h in range(len(explanation.local_exp[i])):
            mask_vls.append(explanation.local_exp[i][h][1])
            mask_vls.append(-explanation.local_exp[i][h][1])
        loc_max = max(mask_vls)
        global_max = max(global_max, loc_max)
    lime_values_list = []
    global_lime_values_list = []
    segments = explanation.segments - np.min(explanation.segments)
    for i in range(6):
        lime_values = np.zeros([64, 64])
        global_lime_values = np.zeros([64, 64])
        mask_vals = {}
        for h in range(len(explanation.local_exp[i])):
            mask_vals[explanation.local_exp[i][h][0]] = explanation.local_exp[i][h][1]
        sh_min = min(mask_vals.values())
        sh_max = max(mask_vals.values())
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = mask_vals[segments[s, t]] / local_max
                lime_values[s, t] = 0.5 + (normd / 2)
                normg = mask_vals[segments[s, t]] / global_max
                global_lime_values[s, t] = 0.5 + (normg / 2)
        lime_values_list.append(lime_values)
        global_lime_values_list.append(global_lime_values)

    return lime_values_list, global_lime_values_list, greens

def lime_8_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet8, 'weights/new_classify2d_8_outs.ckpt', device='cpu')
    classifier_wrapper.fit()
    segmenter = MySegmenter("felzenszwalb", 200, 0.5, 50)
    segmenter_with_mask = LungSegmenter(segmenter)

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        sample_to_explain,
        classifier_wrapper.predict_proba,
        top_labels=8,
        hide_color=[55, 55, 55],
        num_samples=10000,
        segmentation_fn=segmenter_with_mask
    )
    global_max = 0
    for i in range(8):
        mask_vls = []
        for h in range(len(explanation.local_exp[i])):
            mask_vls.append(explanation.local_exp[i][h][1])
            mask_vls.append(-explanation.local_exp[i][h][1])
        loc_max = max(mask_vls)
        global_max = max(global_max, loc_max)
    lime_values_list = []
    global_lime_values_list = []
    segments = explanation.segments - np.min(explanation.segments)
    for i in range(8):
        lime_values = np.zeros([64, 64])
        global_lime_values = np.zeros([64, 64])
        mask_vals = {}
        for h in range(len(explanation.local_exp[i])):
            mask_vals[explanation.local_exp[i][h][0]] = explanation.local_exp[i][h][1]
        sh_min = min(mask_vals.values())
        sh_max = max(mask_vals.values())
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = mask_vals[segments[s, t]] / local_max
                lime_values[s, t] = 0.5 + (normd / 2)
                normg = mask_vals[segments[s, t]] / global_max
                global_lime_values[s, t] = 0.5 + (normg / 2)
        lime_values_list.append(lime_values)
        global_lime_values_list.append(global_lime_values)

    return lime_values_list, global_lime_values_list, greens

def shap_4_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet4, 'weights/new_classify2d_4_outs.ckpt', device='cpu')
    classifier_wrapper.fit()

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    sample_to_explain = torch.tensor(sample_to_explain)

    masker = shap.maskers.Image("inpaint_ns", sample_to_explain.shape)

    explainer = shap.Explainer(classifier_wrapper.predict_proba,
                               masker,
                               algorithm="partition",
                               output_names=[i for i in range(4)],
                               )

    topk = 4
    batch_size = 50
    n_evals = 10000

    shap_values = explainer(
        sample_to_explain.unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    segregated = [shap_values.output_names.index(k) for k in range(4)]
    global_max = 0
    for i in range(4):
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        loc_max = max(-sh_min, sh_max)
        global_max = max(global_max, loc_max)
    shap_values_list = []
    global_shap_values_list = []
    for i in range(4):
        shap_vals = np.zeros([64, 64])
        global_shap_vals = np.zeros([64, 64])
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = shap_values.values[0, s, t, 0, segregated[i]] / local_max
                shap_vals[s, t] = 0.5 + (normd / 2)
                normg = shap_values.values[0, s, t, 0, segregated[i]] / global_max
                global_shap_vals[s, t] = 0.5 + (normg / 2)
        shap_values_list.append(shap_vals)
        global_shap_values_list.append(global_shap_vals)
    return shap_values_list, global_shap_values_list, greens

def shap_16_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(ClassifyNet16, 'weights/old_classify2d_16_outs.ckpt', device='cpu')
    classifier_wrapper.fit()

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    sample_to_explain = torch.tensor(sample_to_explain)

    masker = shap.maskers.Image("inpaint_ns", sample_to_explain.shape)

    explainer = shap.Explainer(classifier_wrapper.predict_proba,
                               masker,
                               algorithm="partition",
                               output_names=[i for i in range(16)],
                               )

    topk = 16
    batch_size = 50
    n_evals = 10000

    shap_values = explainer(
        sample_to_explain.unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    segregated = [shap_values.output_names.index(k) for k in range(16)]
    global_max = 0
    for i in range(16):
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        loc_max = max(-sh_min, sh_max)
        global_max = max(global_max, loc_max)
    shap_values_list = []
    global_shap_values_list = []
    for i in range(16):
        shap_vals = np.zeros([64, 64])
        global_shap_vals = np.zeros([64, 64])
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = shap_values.values[0, s, t, 0, segregated[i]] / local_max
                shap_vals[s, t] = 0.5 + (normd / 2)
                normg = shap_values.values[0, s, t, 0, segregated[i]] / global_max
                global_shap_vals[s, t] = 0.5 + (normg / 2)
        shap_values_list.append(shap_vals)
        global_shap_values_list.append(global_shap_vals)
    return shap_values_list, global_shap_values_list, greens

def shap_5_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet, 'weights/new_classify2d.ckpt', device='cpu')
    classifier_wrapper.fit()

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    sample_to_explain = torch.tensor(sample_to_explain)

    masker = shap.maskers.Image("inpaint_ns", sample_to_explain.shape)

    explainer = shap.Explainer(classifier_wrapper.predict_proba,
                               masker,
                               algorithm="partition",
                               output_names=[i for i in range(5)],
                               )

    topk = 5
    batch_size = 50
    n_evals = 10000

    shap_values = explainer(
        sample_to_explain.unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    segregated = [shap_values.output_names.index(k) for k in range(5)]
    global_max = 0
    for i in range(4):
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        loc_max = max(-sh_min, sh_max)
        global_max = max(global_max, loc_max)
    shap_values_list = []
    global_shap_values_list = []
    for i in range(4):
        shap_vals = np.zeros([64, 64])
        global_shap_vals = np.zeros([64, 64])
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = shap_values.values[0, s, t, 0, segregated[i]] / local_max
                shap_vals[s, t] = 0.5 + (normd / 2)
                normg = shap_values.values[0, s, t, 0, segregated[i]] / global_max
                global_shap_vals[s, t] = 0.5 + (normg / 2)
        shap_values_list.append(shap_vals)
        global_shap_values_list.append(global_shap_vals)
    return shap_values_list, global_shap_values_list, greens

def shap_17_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(ClassifyNet, 'weights/classify2d.ckpt', device='cpu')
    classifier_wrapper.fit()

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    sample_to_explain = torch.tensor(sample_to_explain)

    masker = shap.maskers.Image("inpaint_ns", sample_to_explain.shape)

    explainer = shap.Explainer(classifier_wrapper.predict_proba,
                               masker,
                               algorithm="partition",
                               output_names=[i for i in range(17)],
                               )

    topk = 17
    batch_size = 50
    n_evals = 10000

    shap_values = explainer(
        sample_to_explain.unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    segregated = [shap_values.output_names.index(k) for k in range(17)]
    global_max = 0
    for i in range(16):
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        loc_max = max(-sh_min, sh_max)
        global_max = max(global_max, loc_max)
    shap_values_list = []
    global_shap_values_list = []
    for i in range(16):
        shap_vals = np.zeros([64, 64])
        global_shap_vals = np.zeros([64, 64])
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = shap_values.values[0, s, t, 0, segregated[i]] / local_max
                shap_vals[s, t] = 0.5 + (normd / 2)
                normg = shap_values.values[0, s, t, 0, segregated[i]] / global_max
                global_shap_vals[s, t] = 0.5 + (normg / 2)
        shap_values_list.append(shap_vals)
        global_shap_values_list.append(global_shap_vals)
    return shap_values_list, global_shap_values_list, greens

def shap_6_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet6, 'weights/newest_classify2d_6_outs.ckpt', device='cpu')
    classifier_wrapper.fit()

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    sample_to_explain = torch.tensor(sample_to_explain)

    masker = shap.maskers.Image("inpaint_ns", sample_to_explain.shape)

    explainer = shap.Explainer(classifier_wrapper.predict_proba,
                               masker,
                               algorithm="partition",
                               output_names=[i for i in range(6)],
                               )

    topk = 6
    batch_size = 50
    n_evals = 10000

    shap_values = explainer(
        sample_to_explain.unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    segregated = [shap_values.output_names.index(k) for k in range(6)]
    global_max = 0
    for i in range(6):
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        loc_max = max(-sh_min, sh_max)
        global_max = max(global_max, loc_max)
    shap_values_list = []
    global_shap_values_list = []
    for i in range(6):
        shap_vals = np.zeros([64, 64])
        global_shap_vals = np.zeros([64, 64])
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = shap_values.values[0, s, t, 0, segregated[i]] / local_max
                shap_vals[s, t] = 0.5 + (normd / 2)
                normg = shap_values.values[0, s, t, 0, segregated[i]] / global_max
                global_shap_vals[s, t] = 0.5 + (normg / 2)
        shap_values_list.append(shap_vals)
        global_shap_values_list.append(global_shap_vals)
    return shap_values_list, global_shap_values_list, greens

def shap_6_ones_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet6, 'weights/newest_classify2d_ones_6_outs.ckpt', device='cpu')
    classifier_wrapper.fit()

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    sample_to_explain = torch.tensor(sample_to_explain)

    masker = shap.maskers.Image("inpaint_ns", sample_to_explain.shape)

    explainer = shap.Explainer(classifier_wrapper.predict_proba,
                               masker,
                               algorithm="partition",
                               output_names=[i for i in range(6)],
                               )

    topk = 6
    batch_size = 50
    n_evals = 10000

    shap_values = explainer(
        sample_to_explain.unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    segregated = [shap_values.output_names.index(k) for k in range(6)]
    global_max = 0
    for i in range(6):
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        loc_max = max(-sh_min, sh_max)
        global_max = max(global_max, loc_max)
    shap_values_list = []
    global_shap_values_list = []
    for i in range(6):
        shap_vals = np.zeros([64, 64])
        global_shap_vals = np.zeros([64, 64])
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = shap_values.values[0, s, t, 0, segregated[i]] / local_max
                shap_vals[s, t] = 0.5 + (normd / 2)
                normg = shap_values.values[0, s, t, 0, segregated[i]] / global_max
                global_shap_vals[s, t] = 0.5 + (normg / 2)
        shap_values_list.append(shap_vals)
        global_shap_values_list.append(global_shap_vals)
    return shap_values_list, global_shap_values_list, greens

def shap_8_outs_explainer(sample_to_explain):
    classifier_wrapper = TorchModelWrapper(NewClassifyNet8, 'weights/new_classify2d_8_outs.ckpt', device='cpu')
    classifier_wrapper.fit()

    predicted_class = classifier_wrapper.predict_proba([sample_to_explain])
    classes = predicted_class > 0.35
    greens = []
    for i in range(len(classes[0])):
        if classes[0, i]:
            greens.append(i)

    sample_to_explain = torch.tensor(sample_to_explain)

    masker = shap.maskers.Image("inpaint_ns", sample_to_explain.shape)

    explainer = shap.Explainer(classifier_wrapper.predict_proba,
                               masker,
                               algorithm="partition",
                               output_names=[i for i in range(8)],
                               )

    topk = 8
    batch_size = 50
    n_evals = 10000

    shap_values = explainer(
        sample_to_explain.unsqueeze(0),
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    segregated = [shap_values.output_names.index(k) for k in range(8)]
    global_max = 0
    for i in range(8):
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        loc_max = max(-sh_min, sh_max)
        global_max = max(global_max, loc_max)
    shap_values_list = []
    global_shap_values_list = []
    for i in range(8):
        shap_vals = np.zeros([64, 64])
        global_shap_vals = np.zeros([64, 64])
        sh_min = shap_values.values[0, :, :, 0, segregated[i]].min()
        sh_max = shap_values.values[0, :, :, 0, segregated[i]].max()
        local_max = max(-sh_min, sh_max)
        for s in range(64):
            for t in range(64):
                normd = shap_values.values[0, s, t, 0, segregated[i]] / local_max
                shap_vals[s, t] = 0.5 + (normd / 2)
                normg = shap_values.values[0, s, t, 0, segregated[i]] / global_max
                global_shap_vals[s, t] = 0.5 + (normg / 2)
        shap_values_list.append(shap_vals)
        global_shap_values_list.append(global_shap_vals)
    return shap_values_list, global_shap_values_list, greens


### Displaying functions

def slider_window_4_images(lung_image, local_values, global_values, greens, cmap):
    def crazy_function(array, coefficient):
        if coefficient == 0:
            return array
        else:
            return (1+np.sign(array-0.5)*np.tanh((10*coefficient)**0.7*np.abs(2*(array-0.5))**(1-coefficient))/np.tanh((10*coefficient)**0.7))/2

    def overlay_mask(image, mask, alpha, coefficient):
        recalculated_mask = crazy_function(mask, coefficient)
        color_mask = cmap(recalculated_mask)[:, :, :3]
        return (1 - alpha) * image + alpha * color_mask

    values = global_values

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='black')
    plt.subplots_adjust(bottom=0.25)

    current_state = {"toggle": True}

    row_labels = ["Right Lung", "Left Lung"]
    col_labels = ["Pneumothorax", "Pleural Effusion"]
    im_list = []
    i = 0
    for r in range(2):
        for c in range(2):
            lungs = lung_image.copy()
            mask = values[i].copy()
            im_list.append(axes[r, c].imshow(np.flipud(overlay_mask(lungs, mask, 0.5, 0))))
            if i in greens:
                clr = 'green'
            else:
                clr = 'white'
            axes[r,c].text(
                x=0.5,
                y=0.1,
                s=f"Output {i}",
                fontsize=11,
                color=clr,
                ha='center',
                va='top',
                transform=axes[r,c].transAxes
            )
            if r == 0:
                axes[r, c].set_title(col_labels[c], fontsize=12, color='white')
            if c == 0:
                axes[r, c].set_ylabel(row_labels[r], fontsize=12, rotation=90, color='white')
            i += 1

    ax_button = plt.axes([0.9, 0.05, 0.05, 0.05])
    ax_a = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='dimgray')
    ax_b = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='dimgray')

    toggle_button = Button(ax_button, 'Global', color='dimgray', hovercolor='gray')
    slider_a = Slider(ax_a, "Alpha", 0, 1, valinit=0.5, track_color='gray')
    slider_b = Slider(ax_b, "Coefficient", 0, 1, valinit=0.0, track_color='gray')

    toggle_button.label.set_color('white')
    slider_a.label.set_color('white')
    slider_b.label.set_color('white')
    slider_a.valtext.set_color('white')
    slider_b.valtext.set_color('white')

    def update(val):
        alpha = slider_a.val
        coefficient = slider_b.val
        values = global_values if current_state["toggle"] else local_values
        i = 0
        for r in range(2):
            for c in range(2):
                mask = values[i].copy()
                im_list[i].set_data(np.flipud(overlay_mask(lung_image, mask, alpha, coefficient)))
                i += 1
        fig.canvas.draw_idle()

    def toggle_state(event):
        current_state["toggle"] = not current_state["toggle"]
        label = "Global" if current_state["toggle"] else "Local"
        toggle_button.label.set_text(label)
        update(None)

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    toggle_button.on_clicked(toggle_state)

    fig.text(0.92, 0.12, "Color reference", fontsize=12, color='white', ha='center')

    plt.show()

def slider_window_16_images(lung_image, local_values, global_values, greens, cmap):
    def crazy_function(array, coefficient):
        if coefficient == 0:
            return array
        else:
            return (1+np.sign(array-0.5)*np.tanh((10*coefficient)**0.7*np.abs(2*(array-0.5))**(1-coefficient))/np.tanh((10*coefficient)**0.7))/2

    def overlay_mask(image, mask, alpha, coefficient):
        recalculated_mask = crazy_function(mask, coefficient)
        color_mask = cmap(recalculated_mask)[:, :, :3]
        return (1 - alpha) * image + alpha * color_mask

    values = global_values

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), facecolor='black')
    plt.subplots_adjust(bottom=0.25)

    current_state = {"toggle": True}

    im_list = []
    i = 0
    for r in range(4):
        for c in range(4):
            lungs = lung_image.copy()
            mask = values[i].copy()
            im_list.append(axes[r, c].imshow(np.flipud(overlay_mask(lungs, mask, 0.5, 0))))
            if i in greens:
                clr = 'green'
            else:
                clr = 'white'
            axes[r,c].text(
                x=0.5,
                y=0.1,
                s=f"Output {i}",
                fontsize=7,
                color=clr,
                ha='center',
                va='top',
                transform=axes[r,c].transAxes
            )
            i += 1

    ax_button = plt.axes([0.9, 0.05, 0.05, 0.05])
    ax_a = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='dimgray')
    ax_b = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='dimgray')

    toggle_button = Button(ax_button, 'Global', color='dimgray', hovercolor='gray')
    slider_a = Slider(ax_a, "Alpha", 0, 1, valinit=0.5, track_color='gray')
    slider_b = Slider(ax_b, "Coefficient", 0, 1, valinit=0.0, track_color='gray')

    toggle_button.label.set_color('white')
    slider_a.label.set_color('white')
    slider_b.label.set_color('white')
    slider_a.valtext.set_color('white')
    slider_b.valtext.set_color('white')

    def update(val):
        alpha = slider_a.val
        coefficient = slider_b.val
        values = global_values if current_state["toggle"] else local_values
        i = 0
        for r in range(4):
            for c in range(4):
                mask = values[i].copy()
                im_list[i].set_data(np.flipud(overlay_mask(lung_image, mask, alpha, coefficient)))
                i += 1
        fig.canvas.draw_idle()

    def toggle_state(event):
        current_state["toggle"] = not current_state["toggle"]
        label = "Global" if current_state["toggle"] else "Local"
        toggle_button.label.set_text(label)
        update(None)

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    toggle_button.on_clicked(toggle_state)

    fig.text(0.92, 0.12, "Color reference", fontsize=12, color='white', ha='center')

    plt.show()

def slider_window_6_images(lung_image, local_values, global_values, greens, cmap):
    def crazy_function(array, coefficient):
        if coefficient == 0:
            return array
        else:
            return (1+np.sign(array-0.5)*np.tanh((10*coefficient)**0.7*np.abs(2*(array-0.5))**(1-coefficient))/np.tanh((10*coefficient)**0.7))/2

    def overlay_mask(image, mask, alpha, coefficient):
        recalculated_mask = crazy_function(mask, coefficient)
        color_mask = cmap(recalculated_mask)[:, :, :3]
        return (1 - alpha) * image + alpha * color_mask

    values = global_values

    fig, axes = plt.subplots(2, 3, figsize=(10, 10), facecolor='black')
    plt.subplots_adjust(bottom=0.25)

    current_state = {"toggle": True}

    row_labels = ["Right Lung", "Left Lung"]
    col_labels = ["Pneumothorax", "Pleural Effusion", "Both"]
    im_list = []
    i = 0
    for r in range(2):
        for c in range(3):
            lungs = lung_image.copy()
            mask = values[i].copy()
            im_list.append(axes[r, c].imshow(np.flipud(overlay_mask(lungs, mask, 0.5, 0))))
            if i in greens:
                clr = 'green'
            else:
                clr = 'white'
            axes[r,c].text(
                x=0.5,
                y=0.1,
                s=f"Output {i}",
                fontsize=11,
                color=clr,
                ha='center',
                va='top',
                transform=axes[r,c].transAxes
            )
            if r == 0:
                axes[r, c].set_title(col_labels[c], fontsize=12, color='white')
            if c == 0:
                axes[r, c].set_ylabel(row_labels[r], fontsize=12, rotation=90, color='white')
            i += 1

    ax_button = plt.axes([0.9, 0.05, 0.05, 0.05])
    ax_a = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='dimgray')
    ax_b = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='dimgray')

    toggle_button = Button(ax_button, 'Global', color='dimgray', hovercolor='gray')
    slider_a = Slider(ax_a, "Alpha", 0, 1, valinit=0.5, track_color='gray')
    slider_b = Slider(ax_b, "Coefficient", 0, 1, valinit=0.0, track_color='gray')

    toggle_button.label.set_color('white')
    slider_a.label.set_color('white')
    slider_b.label.set_color('white')
    slider_a.valtext.set_color('white')
    slider_b.valtext.set_color('white')

    def update(val):
        alpha = slider_a.val
        coefficient = slider_b.val
        values = global_values if current_state["toggle"] else local_values
        i = 0
        for r in range(2):
            for c in range(3):
                mask = values[i].copy()
                im_list[i].set_data(np.flipud(overlay_mask(lung_image, mask, alpha, coefficient)))
                i += 1
        fig.canvas.draw_idle()

    def toggle_state(event):
        current_state["toggle"] = not current_state["toggle"]
        label = "Global" if current_state["toggle"] else "Local"
        toggle_button.label.set_text(label)
        update(None)

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    toggle_button.on_clicked(toggle_state)

    fig.text(0.92, 0.12, "Color reference", fontsize=12, color='white', ha='center')

    plt.show()

def slider_window_8_images(lung_image, local_values, global_values, greens, cmap):
    def crazy_function(array, coefficient):
        if coefficient == 0:
            return array
        else:
            return (1+np.sign(array-0.5)*np.tanh((10*coefficient)**0.7*np.abs(2*(array-0.5))**(1-coefficient))/np.tanh((10*coefficient)**0.7))/2

    def overlay_mask(image, mask, alpha, coefficient):
        recalculated_mask = crazy_function(mask, coefficient)
        color_mask = cmap(recalculated_mask)[:, :, :3]
        return (1 - alpha) * image + alpha * color_mask

    values = global_values

    fig, axes = plt.subplots(2, 4, figsize=(10, 10), facecolor='black')
    plt.subplots_adjust(bottom=0.25)

    current_state = {"toggle": True}

    row_labels = ["Right Lung", "Left Lung"]
    col_labels = ["None", "Pneumothorax", "Pleural Effusion", "Both"]
    im_list = []
    i = 0
    for r in range(2):
        for c in range(4):
            lungs = lung_image.copy()
            mask = values[i].copy()
            im_list.append(axes[r, c].imshow(np.flipud(overlay_mask(lungs, mask, 0.5, 0))))
            if i in greens:
                clr = 'green'
            else:
                clr = 'white'
            axes[r,c].text(
                x=0.5,
                y=0.1,
                s=f"Output {i}",
                fontsize=11,
                color=clr,
                ha='center',
                va='top',
                transform=axes[r,c].transAxes
            )
            if r == 0:
                axes[r, c].set_title(col_labels[c], fontsize=12, color='white')
            if c == 0:
                axes[r, c].set_ylabel(row_labels[r], fontsize=12, rotation=90, color='white')
            i += 1

    ax_button = plt.axes([0.9, 0.05, 0.05, 0.05])
    ax_a = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='dimgray')
    ax_b = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor='dimgray')

    toggle_button = Button(ax_button, 'Global', color='dimgray', hovercolor='gray')
    slider_a = Slider(ax_a, "Alpha", 0, 1, valinit=0.5, track_color='gray')
    slider_b = Slider(ax_b, "Coefficient", 0, 1, valinit=0.0, track_color='gray')

    toggle_button.label.set_color('white')
    slider_a.label.set_color('white')
    slider_b.label.set_color('white')
    slider_a.valtext.set_color('white')
    slider_b.valtext.set_color('white')

    def update(val):
        alpha = slider_a.val
        coefficient = slider_b.val
        values = global_values if current_state["toggle"] else local_values
        i = 0
        for r in range(2):
            for c in range(4):
                mask = values[i].copy()
                im_list[i].set_data(np.flipud(overlay_mask(lung_image, mask, alpha, coefficient)))
                i += 1
        fig.canvas.draw_idle()

    def toggle_state(event):
        current_state["toggle"] = not current_state["toggle"]
        label = "Global" if current_state["toggle"] else "Local"
        toggle_button.label.set_text(label)
        update(None)

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    toggle_button.on_clicked(toggle_state)

    fig.text(0.92, 0.12, "Color reference", fontsize=12, color='white', ha='center')

    plt.show()

### Main program

def main():
    ### Ask user for parameters
    NO_SAMPLE = int(input(
        "Please input a sample number you want to explain \nAvailable sample numbers are 0 - 39999 \nSample number: "))
    explainer_type = int(input("Which explainer type do you want to use? \n 1. LIME \n 2. SHAP \nInput number: "))
    explainer_outs = int(input(
        "Which explainer do you want to use? \n 1. 4 class explainer (recommended) \n 2. 16 class explainer \n 3. 5 class explainer \n 4. 17 class explainer \n 5. 6 class explainer \n 6. 6 class explainer (ones) \n 7. 8 class explainer \nInput number: "))
    colors_type = int(input(
        "What colors do u want the explanation to be displayed in? \n 1. Blue-Red scale (recommended) \n 2. I want to use a color from matplotlib.colormaps \nInput number: "))
    if colors_type == 1:
        cmap = PaletteRedBlue()
    elif colors_type == 2:
        color_name = input(
            "Write an exact name of the color palette you want to use \nColor palette name: ")
        cmap = plt.get_cmap(color_name)

    ### Load data
    print("Loading data...")
    data = torch.load('data/test/images_no_elipses.ds')
    labels = torch.load('data/test/labels_no_elipses.ds')
    logits = torch.load('data/test/logits_no_elipses.ds')

    ### to RGB and 0-255 to 0-1 for the image
    sample = data[NO_SAMPLE]
    sample_to_explain = gray2rgb(sample)
    sample_to_display = (sample_to_explain - sample_to_explain.min()) / (
                sample_to_explain.max() - sample_to_explain.min())

    ### Explanation
    if explainer_type == 1:
        print("Creating LIME explanation...")
    else:
        print("Creating SHAP explanation...")
        print("It might take up to a minute")
    if explainer_type == 1 and explainer_outs == 1:
        local_values_list, global_values_list, greens = lime_4_outs_explainer(sample_to_explain)
    elif explainer_type == 1 and explainer_outs == 2:
        local_values_list, global_values_list, greens = lime_16_outs_explainer(sample_to_explain)
    elif explainer_type == 1 and explainer_outs == 3:
        local_values_list, global_values_list, greens = lime_5_outs_explainer(sample_to_explain)
    elif explainer_type == 1 and explainer_outs == 4:
        local_values_list, global_values_list, greens = lime_17_outs_explainer(sample_to_explain)
    elif explainer_type == 1 and explainer_outs == 5:
        local_values_list, global_values_list, greens = lime_6_outs_explainer(sample_to_explain)
    elif explainer_type == 1 and explainer_outs == 6:
        local_values_list, global_values_list, greens = lime_6_ones_outs_explainer(sample_to_explain)
    elif explainer_type == 1 and explainer_outs == 7:
        local_values_list, global_values_list, greens = lime_8_outs_explainer(sample_to_explain)
    elif explainer_type == 2 and explainer_outs == 1:
        local_values_list, global_values_list, greens = shap_4_outs_explainer(sample_to_explain)
    elif explainer_type == 2 and explainer_outs == 2:
        local_values_list, global_values_list, greens = shap_16_outs_explainer(sample_to_explain)
    elif explainer_type == 2 and explainer_outs == 3:
        local_values_list, global_values_list, greens = shap_5_outs_explainer(sample_to_explain)
    elif explainer_type == 2 and explainer_outs == 4:
        local_values_list, global_values_list, greens = shap_17_outs_explainer(sample_to_explain)
    elif explainer_type == 2 and explainer_outs == 5:
        local_values_list, global_values_list, greens = shap_6_outs_explainer(sample_to_explain)
    elif explainer_type == 2 and explainer_outs == 6:
        local_values_list, global_values_list, greens = shap_6_ones_outs_explainer(sample_to_explain)
    elif explainer_type == 2 and explainer_outs == 7:
        local_values_list, global_values_list, greens = shap_8_outs_explainer(sample_to_explain)

    # Display
    if explainer_outs == 5:
        slider_window_6_images(sample_to_display, local_values_list, global_values_list, greens, cmap)
    elif explainer_outs == 6:
        slider_window_6_images(sample_to_display, local_values_list, global_values_list, greens, cmap)
    elif explainer_outs == 7:
        slider_window_8_images(sample_to_display, local_values_list, global_values_list, greens, cmap)
    elif explainer_outs%2 == 0:
        slider_window_16_images(sample_to_display, local_values_list, global_values_list, greens, cmap)
    else:
        slider_window_4_images(sample_to_display, local_values_list, global_values_list, greens, cmap)


if __name__ == "__main__":
    main()
