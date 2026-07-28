# FitMeasure AI — Team Operating System

Prep doc for the **internal team meeting**, ahead of the one-on-one coaching session.

**Coach's instruction:** before the one-on-one, hold an internal team meeting to agree on
**tasks, functions, roles, and our "operating system"** — who owns what, and how we work
together. Bring the results to the coaching session to review.

> **Status:** roles, area ownership, tie-breakers, cadence, decision rules and workload are
> **agreed**. Please read §2 and §3 before the meeting and flag anything you disagree with.

---

## 1. Internal meeting agenda (~90 min)

| # | Block | Time | Output |
|---|---|---|---|
| 1 | Where we are — MVP status, what's blocking | 10 min | Shared picture |
| 2 | Roles & functions — confirm who owns what (§2) | 25 min | Ownership confirmed |
| 3 | Operating system — confirm how we work (§3) | 25 min | Cadence + decision rules confirmed |
| 4 | Questions for the coach (§4) | 15 min | Final list |

---

## 2. Roles, functions & ownership

Each area needs **one accountable owner** (the person who answers for it), not a committee.

| Person | Role | Owns (accountable) | Supports |
|---|---|---|---|
| **Maryna Molchanova** | Supervisor · PhD, Senior Lecturer | Academic supervision, methodology · Pilot / customer outreach · Presentations & program deliverables | All areas |
| **Kostiantyn Lianskorunskyi** | Team Lead · PM · ML | Roadmap & priorities · External comms · Backend / API & uptime | Core ML pipeline (review-only) · Presentations (review-only) |
| **Maksim Shurypa** | ML · Developer | Core ML pipeline / measurement accuracy · Presentations | Privacy & E2E (backup) |
| **Yaroslav Kazmirchuk** | Frontend · ML | UI/UX & photo-capture flow · Frontend / web app | — |
| **Maksym Mykytiuk** | Data Analytics · ML | Business model, pricing, market research · Ground-truth data collection & validation | — |
| **Daria Kryva** | UI/UX · Frontend | UI/UX & photo-capture flow · Frontend / web app | — |
| **Vladyslav Murava** | Data Analytics · ML | Privacy & E2E encryption · Backend / API & uptime · Ground-truth data collection & validation | — |

### Area ownership

| Area | Owner(s) |
|---|---|
| Core ML pipeline / measurement accuracy | **Maksim Shurypa** (Kostiantyn: review-only) |
| Frontend / web app | **Daria Kryva** + **Yaroslav Kazmirchuk** |
| UI/UX & photo-capture flow | **Daria Kryva** + **Yaroslav Kazmirchuk** |
| Backend / API & uptime | **Vladyslav Murava** + **Kostiantyn Lianskorunskyi** |
| Privacy & E2E encryption | **Vladyslav Murava** (backup: **Maksim Shurypa**) |
| Business model, pricing, market research | **Maksym Mykytiuk** |
| Presentations, pitch & program deliverables | **Maryna Molchanova** + **Maksim Shurypa** (Kostiantyn: review-only) |
| Ground-truth data collection & validation | **Maksym Mykytiuk** + **Vladyslav Murava** |
| Pilot / customer outreach (local brands & ateliers) | **Maryna Molchanova** |

**Every area has an owner, and every shared area has a tie-breaker.**

### Final say (shared areas)

| Area | Final say |
|---|---|
| Core ML pipeline / measurement accuracy | **Maksim Shurypa** |
| Ground-truth data collection & validation | **Maksym Mykytiuk** |
| Frontend / web app · UI/UX & photo-capture flow | **Daria Kryva** |
| Backend / API & uptime | **Kostiantyn Lianskorunskyi** |
| Presentations, pitch & program deliverables | **Maryna Molchanova** |

Rule: pairs work the area together; if they can't agree, the person above decides and we move on.

---

## 3. Our operating system — how we work

### Cadence
- **Weekly sync:** 1–2 times a week, **30 min**, on **Thursday and Saturday** — the second one
  happens only if the agenda needs it.
- **Async:** everything else runs online, asynchronously.
- **Sprint length:** **2 weeks** on average.

### Decisions
- **Who decides what?** **Technical calls → the area's final-say owner** (see table above);
  **product/priority calls → team lead**; **academic/methodology → supervisor**.
- **Resolving disagreement:** the **field owner decides.** In most cases a result that proves
  itself is **auto-approved** — e.g. *"accuracy of X increased by Y"*. The owner explains it to
  the team; **if it doesn't touch another field, it's approved.**
- **What one person can ship alone:** **bug fixes.**

### Communication
- **Main and urgent channel:** messengers (same channel for both).
- **Response time:** **2–3 hours**, or an agreed time window depending on personal schedules.

### Workload reality check
- Everyone is a student — realistic capacity is **~10 hours per week per person.**
- **Exam periods:** not a problem — only a few days of reduced availability.

---

## 4. Questions for the tutor / coach

> ### ★ Top 3 — ask these first
>
> 1. **Role clarity vs. flexibility.** We're 6 students who all partly overlap (most of us do
>    ML). Should we enforce **strict role boundaries**, or keep overlapping ownership so people
>    can cover for each other? **Where's the line?**
> 2. **Measuring progress.** What should we track **weekly** so we know we're moving — and not
>    just busy? What are good early-stage team KPIs?
> 3. **Right amount of process.** For a team of 6 part-time students (~10 h/week each), how much
>    process is healthy — daily standups, weekly syncs, written specs? **Where does process
>    start costing more than it gives?**

### Full list

### Team building
1. **Role clarity vs. flexibility.** We're 6 students who all partly overlap (most of us do ML).
   Should we enforce **strict role boundaries**, or keep overlapping ownership so people can
   cover for each other? Where's the line?
2. **Motivation without salary or equity.** How do you keep a student team committed to a
   project long-term when there's no pay yet? What actually works — equity split, formal
   commitments, public recognition?
3. **Formalising commitment.** Should a student team formalise long-term ownership and
   contribution **before there's any revenue** — and how? What should the team agree on in
   advance for the case where someone leaves mid-way?
4. **Adding people.** We're missing dedicated **business/sales** capability. Is it better to
   grow the team, or for an existing member to take that on?

### Team management
5. **Right amount of process.** For a team of 6 part-time students, how much process is
   healthy — daily standups, weekly syncs, written specs? Where does process start costing
   more than it gives?
6. **Decision-making model.** For a team our size, do you recommend **single owner decides**,
   or consensus? How does a team keep decisions fast without creating bottlenecks?
7. **Measuring progress.** What should we track weekly so we know we're moving — and not just
   busy? What are good early-stage team KPIs?
8. **Handling underperformance.** In a student team with no formal authority, how do you deal
   with someone consistently not delivering, without breaking the relationship?
9. **Balancing study and startup.** How do teams like ours realistically survive exam periods
   without losing momentum?
10. **Cross-cutting work.** Work that belongs to no single area — planning, external
    communication, admin, program deliverables — tends to pile onto whoever picks it up first.
    How should a startup team distribute that fairly?
