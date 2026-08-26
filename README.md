
# 🤖 What is YOLO?

**YOLO = You Only Look Once.**

It's an **object detection AI model**.

The easiest way to understand it:

### Normal image classification

You give AI:

📷 Road image

AI says:

> **Pothole detected — 94%**

But it doesn't necessarily tell you **where** the pothole is.

---

### YOLO

You give YOLO:

📷 Road image

It says:

> **Pothole — 94%**

AND draws a box around it:

```text
┌──────────────────────────────┐
│                              │
│       ROAD                   │
│                              │
│          ┌──────────┐        │
│          │ POTHOLE  │        │
│          │   94%    │        │
│          └──────────┘        │
│                              │
└──────────────────────────────┘
```

That's **object detection**.

---

# 🧠 What does YOLO actually learn?

Suppose we give it thousands of images:

```text
Image 1 → 🕳️
Image 2 → 🕳️
Image 3 → 🕳️
Image 4 → 🕳️
...
```

But we don't just tell it:

> "This image has a pothole."

We tell it:

> **"This exact region of the image contains a pothole."**

Those rectangles are called **bounding boxes**.

So during training, YOLO learns patterns such as:

* irregular road depressions
* dark regions
* edges
* shapes
* textures
* surrounding road patterns

Eventually it learns:

> "This visual pattern looks like a pothole."

---

# 🎯 What happens when we give it a new image?

Suppose you take this:

📷

```text
Road photo
     ↓
   YOLO
     ↓
┌─────────────────────────────┐
│ Pothole #1 → 96%            │
│ Pothole #2 → 89%            │
│                             │
│ Crack → 81%                 │
└─────────────────────────────┘
```

YOLO returns several pieces of information:

### 1. **Class**

What did it detect?

> `pothole`

### 2. **Confidence**

How confident is the model?

> `0.96` → 96%

### 3. **Bounding box**

Where is it?

Something like:

```text
x1, y1, x2, y2
```

That tells us the coordinates of the rectangle.

---

# 🚀 Why is YOLO good for CivicFix?

Because we don't just want:

> ❌ "There is a pothole."

We want:

> ✅ "There are **2 potholes**, they're **here**, and they're **this large**."

That lets us build additional logic.

For example:

### Small pothole

```text
Area = 2%
→ Low severity
```

### Medium pothole

```text
Area = 8%
→ Medium severity
```

### Large pothole

```text
Area = 25%
→ High severity
```

So:

**YOLO detection**

↓

**Our severity algorithm**

↓

**CivicFix priority**

↓

**Automatic complaint**

That's where our project becomes more than just an AI model.

---

# 🧩 YOLO vs the whole AI system

This distinction is VERY important.

YOLO is **not our entire project**.

Think of it like this:

```text
                 CIVICFIX AI
                      │
          ┌───────────┴───────────┐
          │                       │
      AI MODEL                APPLICATION
          │                       │
        YOLO                 CivicFix
          │                       │
    Detect pothole          Store complaint
          │                 Show dashboard
          ↓                 Track status
    Confidence
    Bounding box
          │
          ↓
    Severity Logic
          │
          ↓
      Priority
```

**YOLO = the brain that sees the pothole.**

**CivicFix = the system that does something about it.**

---

# 👀 And why "You Only Look Once"?

Older object-detection approaches could process an image in multiple stages.

YOLO was designed to look at the image **in one forward pass** and simultaneously predict:

> **What objects are present + where they are.**

That's why it's called:

**You Only Look Once.**

It's also why YOLO became famous for **real-time object detection**.

---

# 🏆 For our competition

We're NOT going to say:

> "We used YOLO because it's popular."

We're going to demonstrate:

**📷 Upload road image**

↓

**🤖 YOLO detects potholes**

↓

**📊 Calculate severity**

↓

**⚠️ Assign priority**

↓

**📝 Create CivicFix complaint**

↓

**📍 Put it on civic dashboard**

